"""A bounded, explicit transition table, rather than a combinatorial search."""
from __future__ import annotations

import importlib.util
import platform
import shutil
from pathlib import Path

import torch
from torch import nn

from kernelixx.optimizations import UnsupportedOptimization

PIPELINES = ("fp16", "onnx", "compile", "quantization", "fp16+compile", "onnx+quantization")


def check_compatibility(pipeline: str, model: nn.Module, device: torch.device) -> None:
    if pipeline not in PIPELINES:
        raise UnsupportedOptimization(f"Unsupported backend/precision transition: {pipeline}")
    stages = pipeline.split("+")
    if "fp16" in stages:
        if device.type != "cuda":
            raise UnsupportedOptimization("FP16 requires CUDA")
        if torch.cuda.get_device_capability(device) < (5, 3):
            raise UnsupportedOptimization("FP16 requires CUDA compute capability >= 5.3")
    if "onnx" in stages:
        if device.type != "cpu":
            raise UnsupportedOptimization("ONNX Runtime adapter is CPU-only")
        if any(importlib.util.find_spec(p) is None for p in ("onnx", "onnxruntime")):
            raise UnsupportedOptimization("Install kernelixx[onnx]")
    if "quantization" in stages:
        if device.type != "cpu":
            raise UnsupportedOptimization("Dynamic INT8 adapter requires CPU")
        if not any(isinstance(m, nn.Linear) for m in model.modules()):
            raise UnsupportedOptimization("No nn.Linear layers eligible for dynamic INT8 quantization")
    if "compile" in stages:
        if not hasattr(torch, "compile"):
            raise UnsupportedOptimization("torch.compile is unavailable")
        if device.type == "cuda" and importlib.util.find_spec("triton") is None:
            raise UnsupportedOptimization("CUDA Inductor requires a compatible Triton installation")
        if device.type == "cpu" and platform.system() == "Windows" and shutil.which("cl") is None:
            raise UnsupportedOptimization("Windows CPU Inductor requires cl.exe")


def compose(pipeline: str, model: nn.Module, sample: torch.Tensor, device: torch.device,
            directory: Path, threads: int):
    # Caller owns an independent model; no extra deep copy inside transforms.
    from kernelixx.optimizations.fusion import compile_model
    from kernelixx.optimizations.precision import fp16
    from kernelixx.optimizations.graph import OnnxRunner, export_onnx
    if pipeline == "fp16+compile":
        runner = fp16(model, device)
        return compile_model(runner.model, device, "fp16")
    if pipeline == "onnx+quantization":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx
        directory.mkdir(parents=True, exist_ok=True)
        runner, details = export_onnx(model, sample, device, directory, threads)
        source = Path(details["export_path"])
        target = directory / "model.int8.onnx"
        del runner
        quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm"], per_channel=True,
                         reduce_range=True)
        onnx.checker.check_model(str(target))
        graph = onnx.load(str(target))
        if not any(n.op_type in {"MatMulInteger", "DynamicQuantizeLinear"} for n in graph.graph.node):
            raise UnsupportedOptimization("ONNX graph has no converted dynamic Linear operations")
        runner = OnnxRunner(target, threads)
        runner.precision = "int8-dynamic-linear"
        return runner, {**details, "export_path": str(target), "per_channel": True, "reduce_range": True,
                        "scope": "Dynamic MatMul/Gemm INT8; other operations unchanged; reduced weight range avoids U8S8 saturation on older x86"}
    raise UnsupportedOptimization(f"Unsupported composition: {pipeline}")

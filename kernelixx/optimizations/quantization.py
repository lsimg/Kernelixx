from __future__ import annotations

import torch
from torch import nn

from kernelixx.core.benchmark import TorchRunner
from kernelixx.optimizations import UnsupportedOptimization


def quantize(model: nn.Module, device: torch.device, *, owned: bool = False) -> tuple[TorchRunner, dict[str, object]]:
    if device.type != "cpu":
        raise UnsupportedOptimization("PyTorch dynamic INT8 Linear quantization uses a CPU backend")
    layers = sum(isinstance(module, nn.Linear) for module in model.modules())
    if not layers:
        raise UnsupportedOptimization("No nn.Linear layers eligible for dynamic INT8 quantization")
    engines = torch.backends.quantized.supported_engines
    engine = next((e for e in ("x86", "fbgemm", "onednn", "qnnpack") if e in engines), None)
    if engine is None:
        raise UnsupportedOptimization("No supported quantized CPU engine is installed")
    previous = torch.backends.quantized.engine
    try:
        torch.backends.quantized.engine = engine
        result = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=owned)
        runner = TorchRunner(result, device, "int8-dynamic-linear")
    finally:
        torch.backends.quantized.engine = previous
    return runner, {"quantized_linear_layers": layers, "engine": engine,
                    "scope": "Only Linear layers; ResNet convolution weights remain FP32."}

from __future__ import annotations

import importlib.util
import platform
import shutil

import torch
from torch import nn

from kernelixx.core.benchmark import TorchRunner
from kernelixx.optimizations import UnsupportedOptimization


def compile_model(model: nn.Module, device: torch.device, precision: str = "fp32") -> tuple[TorchRunner, dict[str, object]]:
    if not hasattr(torch, "compile"):
        raise UnsupportedOptimization("This PyTorch version has no torch.compile")
    if device.type == "cuda" and importlib.util.find_spec("triton") is None:
        raise UnsupportedOptimization("Inductor CUDA requires a compatible Triton installation; absent here")
    if platform.system() == "Windows" and device.type == "cpu" and shutil.which("cl") is None:
        raise UnsupportedOptimization("Inductor CPU on Windows requires an available MSVC C++ toolchain (cl.exe)")
    runner = TorchRunner(model, device, precision)
    runner.model = torch.compile(runner.model, backend="inductor")
    runner.backend = "torch.compile/inductor"
    return runner, {"fusion_confirmed": False,
                    "note": "Inductor requested. No kernel fusion claim without compiler evidence; compilation occurs before timed repetitions."}

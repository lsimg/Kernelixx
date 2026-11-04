from __future__ import annotations

import torch
from torch import nn

from kernelixx.core.benchmark import TorchRunner
from kernelixx.optimizations import UnsupportedOptimization


def fp16(model: nn.Module, device: torch.device) -> TorchRunner:
    if device.type != "cuda":
        raise UnsupportedOptimization("FP16 inference requires CUDA; CPU FP16 is deliberately unsupported")
    if torch.cuda.get_device_capability(device) < (5, 3):
        raise UnsupportedOptimization("GPU compute capability is below the FP16 requirement (5.3)")
    # Individual unsupported operators fail during the explicit validation forward pass.
    return TorchRunner(model.half(), device, "fp16")

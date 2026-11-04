from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn


def load_model(name: str, weights: str = "none") -> tuple[nn.Module, dict[str, Any]]:
    """Custom models use `package.module:factory`; factory returns a CPU nn.Module.

    Custom Python is trusted user code. No checkpoint pickle is loaded by this API.
    """
    if weights not in {"none", "default"}:
        raise ValueError("Weights must be none or default")
    if name in {"resnet18", "resnet50"}:
        from torchvision import models

        enum = models.ResNet18_Weights if name == "resnet18" else models.ResNet50_Weights
        model = getattr(models, name)(weights=enum.DEFAULT if weights == "default" else None)
    elif ":" in name:
        if weights != "none":
            raise ValueError("Custom factories own weight loading; use --weights none")
        module, attribute = name.rsplit(":", 1)
        model = getattr(importlib.import_module(module), attribute)()
    else:
        raise ValueError("Use resnet18, resnet50, or package.module:factory")
    if not isinstance(model, nn.Module):
        raise TypeError("Model factory must return torch.nn.Module")
    model = model.cpu().float().eval()
    info = {"name": name, "architecture": type(model).__name__, "weights": weights,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "architecture_repr": repr(model)}
    return model, info

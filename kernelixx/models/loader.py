from __future__ import annotations

import importlib
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


def source_fingerprint(model: nn.Module, name: str) -> dict[str, Any]:
    """Hash Kernelixx and direct Python definitions, not the dependency closure."""
    modules = {type(component).__module__ for component in model.modules()}
    if ":" in name:
        modules.add(name.rsplit(":", 1)[0])
    sources, unavailable = {}, []
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(package).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        key = ".".join(("kernelixx", *parts))
        try:
            with path.open("rb") as stream:
                sources[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            unavailable.append(key)
    if not package.is_dir():
        unavailable.append("kernelixx:<source tree>")
    for module_name in sorted(modules):
        module = sys.modules.get(module_name)
        try:
            filename = inspect.getsourcefile(module) if module is not None else None
            if filename is None:
                raise OSError("No Python source file")
            with Path(filename).open("rb") as stream:
                sources[module_name] = hashlib.file_digest(stream, "sha256").hexdigest()
        except (OSError, TypeError):
            unavailable.append(module_name)
    payload = {"modules": sources, "unavailable": unavailable}
    return {**payload, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "complete": not unavailable,
            "scope": "Kernelixx Python package plus direct factory/nn.Module class source files; excludes dependency closure, runtime patches and native binaries"}


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            if value.is_quantized:
                digest.update(str(value.qscheme()).encode())
                if value.qscheme() in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
                    visit((value.q_scale(), value.q_zero_point()))
                else:
                    visit((value.q_per_channel_axis(), value.q_per_channel_scales(), value.q_per_channel_zero_points()))
                value = value.int_repr()
            digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast("B"))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list)):
            digest.update(str(type(value)).encode())
            for item in value:
                visit(item)
        elif isinstance(value, (str, int, float, bool, bytes, torch.dtype)) or value is None:
            digest.update(repr(value).encode("utf-8"))
        else:
            raise TypeError(f"Unsupported extra model state for reproducible fingerprint: {type(value).__name__}")
        digest.update(b"\x00")
    for name, value in model.state_dict().items():
        # Preserve the original flat-Tensor fingerprint format so manifests
        # produced before packed-state support remain reloadable.
        digest.update(name.encode())
        if isinstance(value, torch.Tensor) and not value.is_quantized:
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast("B"))
        else:
            visit(value)
    return digest.hexdigest()


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
            "architecture_repr": repr(model), "source_code_fingerprint": source_fingerprint(model, name)}
    return model, info

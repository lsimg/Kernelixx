from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import random
import shutil
import subprocess
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> tuple[torch.device, list[str]]:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    if requested == "cpu":
        return torch.device("cpu"), []
    if torch.cuda.is_available():
        return torch.device("cuda:0"), []
    notes = ["CUDA is unavailable in this PyTorch installation; using CPU."]
    return torch.device("cpu"), notes if requested == "cuda" else []


def seed_everything(seed: int, threads: int = 1) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn rather than silently fail for models without deterministic kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def system_info() -> dict[str, Any]:
    versions = {}
    for package in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "rich"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gpu = {"name": prop.name, "total_memory_bytes": prop.total_memory,
               "compute_capability": f"{prop.major}.{prop.minor}"}
    driver_gpus = []
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            output = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
            for line in output.stdout.strip().splitlines():
                name, driver, memory = (value.strip() for value in line.split(","))
                driver_gpus.append({"name": name, "driver": driver, "total_memory_mib": int(memory)})
        except (OSError, subprocess.SubprocessError, ValueError):
            driver_gpus = []
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "versions": versions,
            "torch_runtime_version": str(torch.__version__), "driver_visible_gpus": driver_gpus,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": gpu, "cpu_threads": torch.get_num_threads(),
            "quantized_engines": list(torch.backends.quantized.supported_engines)}

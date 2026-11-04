from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from kernelixx.utils.device import cleanup, resolve_device, seed_everything, synchronize, system_info


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _train_one(initial: dict[str, torch.Tensor], batches: list[tuple[torch.Tensor, torch.Tensor]],
               device: torch.device, amp: bool, warmup: int) -> dict[str, Any]:
    model = SmallCNN().to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def step(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = nn.functional.cross_entropy(model(inputs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    # Transfers stay outside iteration timing, identical for both variants.
    data = [(x.to(device), y.to(device)) for x, y in batches]
    for i in range(warmup):
        step(*data[i % len(data)])
    synchronize(device)
    # Warmup must not give one variant different initial training weights/state.
    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, losses, scales = [], [], []
    for inputs, labels in data:
        synchronize(device)
        start = time.perf_counter_ns()
        loss = step(inputs, labels)
        synchronize(device)
        times.append((time.perf_counter_ns() - start) / 1e6)
        losses.append(loss.item())
        scales.append(scaler.get_scale())
        del loss
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite training loss")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"method": "amp-fp16" if amp else "fp32", "status": "ok",
            "metrics": {"device": str(device), "latency_median_ms": float(np.median(times)),
                        "latency_p95_ms": float(np.percentile(times, 95)),
                        "throughput_samples_per_sec": sum(x.shape[0] for x, _ in batches) / (sum(times) / 1000),
                        "peak_gpu_memory_bytes": peak},
            "iteration_times_ms": times, "losses": losses, "grad_scaler_scales": scales}


def training_experiment(device_request: str = "auto", seed: int = 42, threads: int = 1,
                        steps: int = 20, warmup: int = 3, batch_size: int = 16) -> dict[str, Any]:
    if steps < 2 or warmup < 1 or batch_size < 1 or threads < 1 or not 0 <= seed < 2**32:
        raise ValueError("steps >= 2, warmup/batch-size/threads >= 1 and seed in [0,2**32) required")
    seed_everything(seed, threads)
    device, warnings = resolve_device(device_request)
    initial = copy.deepcopy(SmallCNN().state_dict())
    generator = torch.Generator().manual_seed(seed + 1)
    batches = [(torch.randn(batch_size, 3, 32, 32, generator=generator),
                torch.randint(0, 10, (batch_size,), generator=generator)) for _ in range(steps)]
    runs: list[dict[str, Any]] = []
    for amp in (False, True):
        if amp and (device.type != "cuda" or torch.cuda.get_device_capability(device) < (5, 3)):
            runs.append({"method": "amp-fp16", "status": "skipped", "reason": "AMP FP16 requires a compatible CUDA GPU"})
            continue
        try:
            seed_everything(seed, threads)
            runs.append(_train_one(initial, batches, device, amp, warmup))
        except Exception as exc:
            runs.append({"method": "amp-fp16" if amp else "fp32", "status": "failed",
                         "reason": f"{type(exc).__name__}: {exc}"})
        finally:
            cleanup(device)
    report: dict[str, Any] = {"schema_version": 1, "experiment": "training", "hardware": system_info(),
                              "config": {"device": device_request, "seed": seed, "threads": threads,
                                         "steps": steps, "warmup": warmup, "batch_size": batch_size,
                                         "input_shape": [batch_size, 3, 32, 32], "optimizer": "SGD lr=0.01"},
                              "runs": runs, "warnings": warnings,
                              "conclusion": "Limited synthetic CNN training experiment; no convergence or task accuracy claim.",
                              "limitations": ["Synthetic random inputs and labels; losses do not measure real task learning.",
                                              "Same initial state and batches; optimizer and scaler reset after warmup.",
                                              "Timing includes forward, backward, optimizer and scaler; excludes data transfer.",
                                              "AMP may skip updates on overflow; scaler trajectories are recorded.",
                                              "CPU peak memory is not measured."]}
    if all(run["status"] == "ok" for run in runs):
        base, opt = (run["metrics"] for run in runs)
        report["comparison"] = {"speedup": base["latency_median_ms"] / opt["latency_median_ms"],
                                "memory_reduction_percent": 100 * (1 - opt["peak_gpu_memory_bytes"] / base["peak_gpu_memory_bytes"])}
    return report

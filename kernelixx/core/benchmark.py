from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from kernelixx.utils.device import synchronize
from kernelixx.core.memory import MemoryMonitor


class Runner(Protocol):
    device: torch.device
    precision: str
    backend: str
    weight_bytes: int
    serialized_bytes: int

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


def tensor_bytes(value: Any) -> int:
    """Include packed quantized tensors in state_dict tuples and dictionaries."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class TorchRunner:
    backend = "pytorch"

    def __init__(self, model: nn.Module, device: torch.device, precision: str = "fp32"):
        self.device, self.precision = device, precision
        self.model = model.eval().to(device)
        state = self.model.state_dict()
        self.weight_bytes = tensor_bytes(state)
        self._serialized_bytes = None

    @property
    def serialized_bytes(self) -> int:
        # Count actual serialization without a model-sized in-memory buffer.
        if self._serialized_bytes is None:
            with tempfile.TemporaryFile() as buffer:
                torch.save(self.model.state_dict(), buffer)
                self._serialized_bytes = buffer.tell()
        return self._serialized_bytes

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 5
    repeats: int = 30

    def __post_init__(self) -> None:
        if self.warmup < 1 or self.repeats < 2:
            raise ValueError("warmup must be >= 1 and repeats >= 2")


def latency_stability(times: list[float]) -> dict[str, Any]:
    if len(times) < 10:
        return {"status": "INSUFFICIENT", "reason": "At least 10 timings are needed for the drift diagnostic"}
    width = len(times) // 3
    first, last = float(np.median(times[:width])), float(np.median(times[-width:]))
    drift = last / first - 1
    return {"status": "STABLE_OBSERVED" if abs(drift) <= 0.1 else "UNSTABLE",
            "first_segment_median_ms": first, "last_segment_median_ms": last,
            "relative_drift": drift, "maximum_allowed_absolute_drift": 0.1,
            "note": "First/last-third median diagnostic, not a stationarity or significance test."}


def measure_cold_start(runner: Runner, inputs: torch.Tensor) -> float:
    with torch.inference_mode():
        synchronize(runner.device)
        start = time.perf_counter_ns()
        output = runner(inputs)
        synchronize(runner.device)
        result = (time.perf_counter_ns() - start) / 1e6
        del output
    return result


def benchmark(runner: Runner, inputs: torch.Tensor, config: BenchmarkConfig,
              *, cold_start_ms: float | None = None) -> dict[str, Any]:
    if inputs.device != runner.device:
        raise ValueError("Inputs and runner must use the same device")
    times: list[float] = []
    total_start = time.perf_counter()
    with torch.inference_mode():
        cold_ms = measure_cold_start(runner, inputs) if cold_start_ms is None else cold_start_ms
        for _ in range(config.warmup):
            output = runner(inputs)
            del output
        synchronize(runner.device)
        start_allocated = None
        if runner.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runner.device)
            start_allocated = torch.cuda.memory_allocated(runner.device)
        for _ in range(config.repeats):
            synchronize(runner.device)
            start = time.perf_counter_ns()
            output = runner(inputs)
            synchronize(runner.device)
            times.append((time.perf_counter_ns() - start) / 1e6)
            del output
        # Observe memory in a separate pass; sampler threads do not perturb latency.
        with MemoryMonitor(runner.device) as memory:
            output = runner(inputs)
            synchronize(runner.device)
            del output
    peak = memory.metrics["torch_peak_allocated_bytes"]
    stability = latency_stability(times)
    return {"device": str(runner.device), "backend": runner.backend,
            "precision": runner.precision, "batch_size": int(inputs.shape[0]),
            "input_shape": list(inputs.shape), "input_dtype": str(inputs.dtype),
            "latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)),
            "throughput_samples_per_sec": inputs.shape[0] * len(times) / (sum(times) / 1000),
            "latencies_ms": times, "peak_gpu_memory_bytes": peak,
            "cold_start_latency_ms": cold_ms,
            "post_warmup_latency_median_ms": float(np.median(times)),
            "steady_state_latency_ms": float(np.median(times)) if stability["status"] == "STABLE_OBSERVED" else None,
            "latency_stability": stability,
            "benchmark_runtime_seconds": time.perf_counter() - total_start,
            "memory": memory.metrics,
            "sampled_peak_ram_bytes": memory.metrics["sampled_process_rss_peak_bytes"],
            "peak_gpu_reserved_bytes": memory.metrics["torch_peak_reserved_bytes"],
            "sampled_process_vram_peak_bytes": memory.metrics["sampled_process_vram_peak_bytes"],
            "gpu_resident_before_inference_bytes": start_allocated,
            "gpu_transient_bytes": peak - start_allocated if peak is not None else None,
            "weight_bytes": runner.weight_bytes, "serialized_model_bytes": runner.serialized_bytes,
            "measurement": asdict(config), "cpu_peak_memory_bytes": None,
            "memory_measurement": "Separate post-warmup inference pass; CPU RSS peak is sampled, not exact."}


def comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparable = all(baseline[key] == candidate[key]
                     for key in ("device", "input_shape", "batch_size", "measurement"))
    bm, cm = baseline["peak_gpu_memory_bytes"], candidate["peak_gpu_memory_bytes"]
    return {"comparable": comparable,
            "speedup": baseline["latency_median_ms"] / candidate["latency_median_ms"] if comparable else None,
            "memory_reduction_percent": 100 * (1 - cm / bm)
            if comparable and bm and cm is not None else None,
            "weight_reduction_percent": 100 * (1 - candidate["weight_bytes"] / baseline["weight_bytes"])
            if baseline["weight_bytes"] else None}

from __future__ import annotations

from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from kernelixx.core.benchmark import TorchRunner
from kernelixx.utils.device import synchronize


def profile_model(runner: TorchRunner, inputs: torch.Tensor) -> dict[str, Any]:
    activities = [ProfilerActivity.CPU]
    if runner.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode():
        runner(inputs)
        synchronize(runner.device)
        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with record_function("kernelixx_inference"):
                runner(inputs)
                synchronize(runner.device)
    events = []
    for event in prof.key_averages():
        if event.key == "kernelixx_inference":
            continue
        events.append({"operator": event.key, "calls": event.count,
                       "self_cpu_time_us": event.self_cpu_time_total,
                       "self_device_time_us": getattr(event, "self_device_time_total", 0.0),
                       "self_cpu_memory_bytes": event.self_cpu_memory_usage})
    metric = "self_device_time_us" if runner.device.type == "cuda" else "self_cpu_time_us"
    events.sort(key=lambda event: event[metric], reverse=True)
    return {"bottlenecks": events[:15], "ranked_by": metric,
            "note": "One profiled inference after warmup; operator self-times are diagnostic, not latency measurements."}

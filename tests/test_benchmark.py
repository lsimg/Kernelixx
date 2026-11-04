import pytest
import torch
from torch import nn

from kernelixx.core.analyzer import profile_model
from kernelixx.core.benchmark import BenchmarkConfig, TorchRunner, benchmark, comparison, tensor_bytes


def test_real_cpu_benchmark_and_sizes():
    model = nn.Linear(16, 4).eval()
    runner = TorchRunner(model, torch.device("cpu"))
    result = benchmark(runner, torch.randn(8, 16), BenchmarkConfig(1, 4))
    assert result["latency_median_ms"] > 0
    assert result["latency_p95_ms"] >= result["latency_median_ms"]
    assert result["throughput_samples_per_sec"] > 0
    assert len(result["latencies_ms"]) == 4
    assert result["peak_gpu_memory_bytes"] is None
    assert result["weight_bytes"] == (16 * 4 + 4) * 4
    assert result["serialized_model_bytes"] > result["weight_bytes"]
    assert comparison(result, result)["speedup"] == 1
    other = {**result, "device": "cuda:0"}
    assert comparison(result, other)["speedup"] is None


def test_packed_tensor_accounting():
    quantized = torch.quantize_per_tensor(torch.randn(2, 3), 0.1, 0, torch.qint8)
    assert tensor_bytes({"packed": (quantized, torch.zeros(2))}) == 6 + 8


@pytest.mark.parametrize("warmup,repeats", [(0, 3), (1, 1)])
def test_invalid_timing(warmup, repeats):
    with pytest.raises(ValueError):
        BenchmarkConfig(warmup, repeats)


def test_profiler_has_real_operator_events():
    profile = profile_model(TorchRunner(nn.Linear(4, 2), torch.device("cpu")), torch.ones(2, 4))
    assert profile["bottlenecks"]
    assert any(event["calls"] > 0 and event["self_cpu_time_us"] > 0 for event in profile["bottlenecks"])

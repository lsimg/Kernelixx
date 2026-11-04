import copy

import numpy as np
import pytest
import torch
from torch import nn

from kernelixx.core.benchmark import BenchmarkConfig, TorchRunner
from kernelixx.core.optimizer import AutoOptimizer, ExperimentConfig, make_batches, select_best
from kernelixx.core.validator import Tolerances
from kernelixx.models.loader import load_model
from kernelixx.optimizations import UnsupportedOptimization
from kernelixx.optimizations.precision import fp16
from kernelixx.optimizations.quantization import quantize
from kernelixx.utils.device import resolve_device


def metric(latency=10, memory=100, device="cpu"):
    return {"device": device, "input_shape": [1, 4], "batch_size": 1,
            "measurement": {"warmup": 1, "repeats": 3}, "latency_median_ms": latency,
            "throughput_samples_per_sec": 1000 / latency, "peak_gpu_memory_bytes": memory,
            "weight_bytes": 100}


def candidate(name, metrics, passed=True, status="ok"):
    return {"method": name, "status": status, "metrics": metrics, "validation": {"passed": passed}}


@pytest.mark.parametrize("objective", ["latency", "throughput", "memory", "balanced"])
def test_selection_uses_measurements_and_validation(objective):
    candidates = [candidate("invalid", metric(1, 1), False), candidate("slower", metric(12, 120)),
                  candidate("good", metric(5, 50)), candidate("cross-device", metric(1, 1, "cuda:0"))]
    result = select_best(metric(), candidates, objective, 0.01)
    assert result["winner"] == "good"
    assert result["improved"]


def test_no_improvement_or_unmeasured_memory():
    assert select_best(metric(), [candidate("slow", metric(11))], "latency", 0.01)["winner"] == "baseline"
    assert select_best(metric(), [candidate("noise", metric(9.99))], "latency", 0.01)["winner"] == "baseline"
    assert not select_best(metric(memory=None), [], "memory", 0.01)["improved"]


def test_seed_reproduces_batches_without_repeating_them():
    config = ExperimentConfig(input_shape=(2, 4))
    first, _ = make_batches(config)
    second, _ = make_batches(config)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[0], first[1])


def test_dataset_contract_and_final_partial_batch(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, inputs=np.ones((5, 4), np.float32), labels=np.zeros(5, np.int64))
    batches, labels = make_batches(ExperimentConfig(input_shape=(2, 4), dataset=str(path)))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert sum(len(batch) for batch in labels) == 5


def test_cuda_fallback_and_fp16_guard(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, notes = resolve_device("cuda")
    assert device.type == "cpu" and notes
    with pytest.raises(UnsupportedOptimization):
        fp16(nn.Linear(4, 2), device)


def test_dynamic_quantization_executes_and_keeps_source():
    source = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).eval()
    runner, details = quantize(copy.deepcopy(source), torch.device("cpu"))
    result = runner(torch.randn(4, 16))
    assert result.shape == (4, 4) and torch.isfinite(result).all()
    assert details["quantized_linear_layers"] == 2
    assert isinstance(source[0], nn.Linear)


def test_no_linear_layers_are_skipped():
    with pytest.raises(UnsupportedOptimization, match="No nn.Linear"):
        quantize(nn.Conv2d(3, 4, 3), torch.device("cpu"))


def test_real_custom_experiment(tmp_path):
    report = AutoOptimizer(ExperimentConfig(model="examples.custom_model:load_model", input_shape=(4, 16),
                                           device="cpu"), BenchmarkConfig(1, 3)).run(tmp_path, ("fp16", "quantization"))
    assert report["baseline"]["latency_median_ms"] > 0
    assert report["candidates"][0]["status"] == "skipped"
    assert report["candidates"][1]["status"] in {"ok", "rejected"}
    assert report["baseline_validation"]["passed"]


def test_candidate_runtime_failure_preserves_baseline(tmp_path, monkeypatch):
    def failing_candidate(*args, **kwargs):
        raise RuntimeError("unsupported operator during export")
    monkeypatch.setattr("kernelixx.core.optimizer.build_candidate", failing_candidate)
    report = AutoOptimizer(ExperimentConfig(model="examples.custom_model:load_model", input_shape=(2, 16),
                                           device="cpu"), BenchmarkConfig(1, 2)).run(tmp_path, ("onnx",))
    assert report["candidates"][0]["status"] == "failed"
    assert "unsupported operator" in report["candidates"][0]["reason"]
    assert report["selection"]["winner"] == "baseline"


def test_accuracy_drop_rejects_numerically_allowed_candidate(tmp_path, monkeypatch):
    torch.manual_seed(42)
    source, _ = load_model("examples.custom_model:load_model")
    x = torch.ones(4, 16)
    with torch.no_grad():
        labels = source(x).argmax(1).numpy()
    dataset = tmp_path / "data.npz"
    np.savez(dataset, inputs=x.numpy(), labels=labels)

    class Inverted(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            return -self.model(inputs)

    def inverted_candidate(method, model, sample, device, directory, threads):
        return TorchRunner(Inverted(model), device), {}

    monkeypatch.setattr("kernelixx.core.optimizer.build_candidate", inverted_candidate)
    report = AutoOptimizer(ExperimentConfig(model="examples.custom_model:load_model", input_shape=(2, 16),
                                           dataset=str(dataset), device="cpu", max_accuracy_drop=0),
                           BenchmarkConfig(1, 2), Tolerances(100, 0)).run(tmp_path, ("onnx",))
    validation = report["candidates"][0]["validation"]
    assert validation["maximum_absolute_error"] < 100
    assert validation["accuracy_drop"] == 1
    assert not validation["passed"]
    assert report["selection"]["winner"] == "baseline"


def test_accuracy_drop_exactly_at_limit_is_allowed(tmp_path, monkeypatch):
    torch.manual_seed(42)
    source, _ = load_model("examples.custom_model:load_model")
    x = torch.ones(100, 16)
    x[0, 0] = 2
    with torch.no_grad():
        labels = source(x).argmax(1).numpy()
    labels[-2:] = (labels[-2:] + 1) % 4
    dataset = tmp_path / "boundary.npz"
    np.savez(dataset, inputs=x.numpy(), labels=labels)

    class OneChangedPrediction(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            logits = self.model(inputs)
            mask = inputs[:, 0] == 2
            logits[mask] = torch.roll(logits[mask], shifts=1, dims=1)
            return logits

    def candidate(method, model, sample, device, directory, threads):
        return TorchRunner(OneChangedPrediction(model), device), {}

    monkeypatch.setattr("kernelixx.core.optimizer.build_candidate", candidate)
    report = AutoOptimizer(ExperimentConfig(model="examples.custom_model:load_model", input_shape=(25, 16),
                                           dataset=str(dataset), device="cpu", max_accuracy_drop=0.01),
                           BenchmarkConfig(1, 2), Tolerances(100, 0)).run(tmp_path, ("onnx",))
    validation = report["candidates"][0]["validation"]
    assert validation["baseline_correct"] == 98
    assert validation["optimized_correct"] == 97
    assert validation["accuracy_drop"] == 0.01
    assert validation["passed"]


@pytest.mark.parametrize("config", [{"threads": 0}, {"seed": -1}, {"input_shape": (0, 3)},
                                    {"min_improvement": 1}, {"validation_batches": 0}])
def test_bad_experiment_config(config):
    with pytest.raises(ValueError):
        ExperimentConfig(**config)


@pytest.mark.parametrize("name,parameters", [("resnet18", 11689512), ("resnet50", 25557032)])
def test_supported_resnets_load(name, parameters):
    model, info = load_model(name)
    assert info["parameters"] == parameters
    assert not model.training
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 1000)

import csv
import copy
import json

import pytest
import torch
from torch import nn

from kernelixx.cli import main
from kernelixx.core.benchmark import BenchmarkConfig, TorchRunner
from kernelixx.core.optimizer import AutoOptimizer, ExperimentConfig
from kernelixx.core.validator import Tolerances, outputs, validate
from kernelixx.optimizations.graph import export_onnx
from kernelixx.optimizations.precision import fp16
from kernelixx.reporting.reporter import create_run_directory, load_report, save_report
from kernelixx.training.benchmark import training_experiment


def test_onnx_real_export_dynamic_batch_and_outputs(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    runner, details = export_onnx(model, torch.randn(2, 4), torch.device("cpu"), tmp_path, 1)
    x = torch.randn(3, 4)
    with torch.inference_mode():
        assert torch.allclose(model(x), runner(x), atol=1e-6)
    assert runner.weight_bytes > 0 and details["opset"] == 17


def test_cpu_training_measures_losses_and_skips_amp():
    report = training_experiment("cpu", steps=3, warmup=1, batch_size=2)
    assert report["runs"][0]["status"] == "ok"
    assert len(report["runs"][0]["losses"]) == 3
    assert report["runs"][0]["metrics"]["latency_median_ms"] > 0
    assert report["runs"][1]["status"] == "skipped"


def test_report_round_trip_and_cli_compare(tmp_path, capsys):
    directory = create_run_directory(tmp_path)
    report = AutoOptimizer(ExperimentConfig(model="examples.custom_model:load_model", input_shape=(2, 16),
                                           device="cpu"), BenchmarkConfig(1, 2)).run(directory, ())
    save_report(report, directory, tmp_path)
    assert load_report(tmp_path / "latest")["baseline"] == report["baseline"]
    assert (directory / "report.md").read_text(encoding="utf-8").startswith("# Kernelixx")
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as source:
        assert list(csv.DictReader(source))[0]["method"] == "baseline"
    assert b"\r\r\n" not in (directory / "metrics.csv").read_bytes()
    main(["compare", "--run-dir", str(directory), "--json"])
    assert json.loads(capsys.readouterr().out)["experiment"] == "inference"


def test_cli_system_json(capsys):
    main(["system-info", "--json"])
    assert "cuda_available" in json.loads(capsys.readouterr().out)


def test_cli_failed_method_has_report_and_nonzero_exit(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["optimize", "--model", "examples.custom_model:load_model", "--input-shape", "2,16",
              "--method", "fp16", "--device", "cpu", "--warmup", "1", "--repeats", "2",
              "--output", str(tmp_path)])
    assert error.value.code == 2
    assert load_report(tmp_path / "latest")["candidates"][0]["status"] == "skipped"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_fp16_and_peak_memory(tmp_path):
    report = AutoOptimizer(ExperimentConfig(model="resnet18", device="cuda", input_shape=(1, 3, 64, 64)),
                           BenchmarkConfig(2, 5)).run(tmp_path, ("fp16",))
    assert report["baseline"]["peak_gpu_memory_bytes"] > report["baseline"]["weight_bytes"]
    candidate = report["candidates"][0]
    # FP16 support does not guarantee this model passes the requested tolerance.
    assert candidate["status"] in {"ok", "rejected"}
    assert candidate["validation"]["finite"]
    assert candidate["validation"]["passed"] == (candidate["status"] == "ok")
    assert candidate["metrics"]["precision"] == "fp16"
    assert candidate["metrics"]["input_dtype"] == "torch.float16"
    assert candidate["metrics"]["peak_gpu_memory_bytes"] < report["baseline"]["peak_gpu_memory_bytes"]
    if not candidate["validation"]["passed"]:
        assert report["selection"]["winner"] == "baseline"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_fp16_exactly_representable_model():
    model = nn.Sequential(nn.Linear(4, 2)).eval()
    with torch.no_grad():
        model[0].weight.fill_(0.125)
        model[0].bias.fill_(0.25)
    device = torch.device("cuda:0")
    batches = [torch.tensor([[1., 2., 3., 4.], [-1., 0., 1., 2.]])]
    reference = outputs(TorchRunner(copy.deepcopy(model), device), batches)
    candidate = outputs(fp16(copy.deepcopy(model), device), batches)
    # These binary fractions and small integers have an exact FP16 result.
    result = validate(reference, candidate, Tolerances(0, 0))
    assert result["passed"]
    assert result["maximum_absolute_error"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_training_executes_both_precisions(tmp_path):
    report = training_experiment("cuda", steps=3, warmup=1, batch_size=2)
    assert all(run["status"] == "ok" for run in report["runs"])
    for run in report["runs"]:
        assert len(run["losses"]) == len(run["grad_scaler_scales"]) == 3
        assert run["metrics"]["peak_gpu_memory_bytes"] > 0
        assert run["metrics"]["latency_median_ms"] > 0
    assert report["comparison"]["speedup"] > 0
    save_report(report, tmp_path, tmp_path)
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as source:
        records = list(csv.DictReader(source))
    measured_ratio = report["runs"][0]["metrics"]["latency_median_ms"] / report["runs"][1]["metrics"]["latency_median_ms"]
    assert float(records[1]["speedup"]) == pytest.approx(measured_ratio)

import copy
import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kernelixx.cli import main, parser
from kernelixx.core.benchmark import BenchmarkConfig, TorchRunner, benchmark, latency_stability
from kernelixx.core.data import BatchSource
from kernelixx.core.export import export_result, reload_artifact
from kernelixx.core.memory import MemoryMonitor
from kernelixx.core.optimizer import AutoOptimizer, ExperimentConfig
from kernelixx.core.pipeline import check_compatibility, compose
from kernelixx.core.repeat import run_repeated, summarize_ratios
from kernelixx.core.validator import Tolerances, ValidationAccumulator, validate
from kernelixx.llm.gguf import GGUFConfig, runtime_parameters
from kernelixx.optimizations import UnsupportedOptimization
from kernelixx.models.loader import state_sha256
from kernelixx.reporting.reporter import save_report


def config(**kwargs):
    return ExperimentConfig(model="examples.custom_model:load_model", device="cpu", input_shape=(2, 16), **kwargs)


def test_streamed_metrics_match_uneven_batches_and_element_weights():
    refs = [torch.randn(3, 7, dtype=torch.float64), torch.randn(1, 7, dtype=torch.float64)]
    outs = [x + 0.125 for x in refs]
    labels = [x.argmax(1) for x in refs]
    result = validate(iter(refs), iter(outs), Tolerances(0.2, 0), iter(labels), chunk_elements=3)
    assert result["passed"] and result["validation_examples"] == 4
    assert result["mean_absolute_error"] == pytest.approx(0.125)
    assert result["class_agreement"] == result["baseline_accuracy"] == result["optimized_accuracy"] == 1
    assert result["accuracy_drop"] == 0
    with pytest.raises(ValueError, match="matching"):
        validate(iter(refs), iter(outs[:1]), Tolerances())


def test_stream_source_npz_npy_order_partial_and_cleanup(tmp_path):
    x, y = np.arange(80, dtype=np.float32).reshape(5, 16), np.arange(5, dtype=np.int64)
    path = tmp_path / "data.npz"
    np.savez_compressed(path, inputs=x, labels=y)
    with BatchSource(config(dataset=str(path))) as source:
        temporary = source.temporary.name
        assert isinstance(source.arrays[0], np.memmap)
        first = list(source)
        assert [len(batch) for batch, _ in first] == [2, 2, 1]
        assert np.array_equal(torch.cat([v for v, _ in first]).numpy(), x)
        assert torch.equal(torch.cat([v for _, v in first]), torch.arange(5))
    from pathlib import Path
    assert not Path(temporary).exists()
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", y)
    with BatchSource(config(dataset=str(tmp_path))) as source:
        assert len(source) == 3
    x[-1, 0] = float("nan")
    np.savez(path, inputs=x, labels=y)
    with pytest.raises(ValueError, match="finite"):
        with BatchSource(config(dataset=str(path))) as source:
            list(source)
    assert not Path(source.temporary.name).exists()


def test_memory_sampler_cleanup_on_failure_and_metric_semantics():
    with pytest.raises(RuntimeError):
        with MemoryMonitor(interval=0.001) as memory:
            allocation = bytearray(4 * 2**20)
            time.sleep(0.01)
            raise RuntimeError("probe")
    assert not memory.thread.is_alive()
    assert memory.metrics["sampled_process_rss_peak_bytes"] >= memory.metrics["process_rss_before_bytes"] > 0
    assert memory.metrics["torch_peak_allocated_bytes"] is None
    assert memory.metrics["sampled_process_vram_peak_bytes"] is None
    del allocation


def test_memory_unavailable_is_null(monkeypatch):
    import builtins
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("test missing optional source")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    with MemoryMonitor() as memory:
        pass
    assert memory.metrics["sampled_process_rss_peak_bytes"] is None
    assert "unavailable" in memory.metrics["notes"][0]


def test_real_latency_memory_and_cold_metrics():
    runner = TorchRunner(nn.Linear(16, 4), torch.device("cpu"))
    assert runner._serialized_bytes is None
    result = benchmark(runner, torch.ones(2, 16), BenchmarkConfig(1, 3))
    assert result["cold_start_latency_ms"] > 0
    assert result["post_warmup_latency_median_ms"] == result["latency_median_ms"]
    assert result["steady_state_latency_ms"] is None
    assert result["latency_stability"]["status"] == "INSUFFICIENT"
    assert result["sampled_peak_ram_bytes"] > 0
    assert result["cpu_peak_memory_bytes"] is None
    assert result["peak_gpu_reserved_bytes"] is None


def test_source_fingerprint_supports_packed_quantized_state():
    source = nn.Sequential(nn.Linear(16, 4)).eval()
    converted = torch.ao.quantization.quantize_dynamic(source, {nn.Linear}, dtype=torch.qint8)
    assert state_sha256(converted) == state_sha256(copy.deepcopy(converted))
    assert state_sha256(source) != state_sha256(converted)


@pytest.mark.parametrize("pipeline,device", [("fp16+onnx", "cpu"), ("onnx+compile", "cpu"),
                                             ("fp16+compile", "cpu"), ("onnx+quantization", "cuda")])
def test_pipeline_incompatible_before_execution(pipeline, device):
    with pytest.raises(UnsupportedOptimization):
        check_compatibility(pipeline, nn.Linear(4, 2), torch.device(device))


def test_real_onnx_int8_chain(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4)).eval()
    x = torch.randn(3, 16)
    check_compatibility("onnx+quantization", model, torch.device("cpu"))
    runner, details = compose("onnx+quantization", model, x, torch.device("cpu"), tmp_path, 1)
    with torch.inference_mode():
        result = validate([model(x)], [runner(x)], Tolerances(0.02, 0.1), classification=False)
    assert result["passed"]
    assert runner.precision == "int8-dynamic-linear"
    assert details["export_path"].endswith(".int8.onnx")


@pytest.mark.parametrize("ratios,expected", [([2, 2], "INCONCLUSIVE"), ([2, 2, 2], "ACCEPT"),
                                            ([1.3, 0.9, 1.2], "INCONCLUSIVE"), ([1, 1, 1], "REJECT")])
def test_repeatability_policy(ratios, expected):
    result = summarize_ratios(ratios, 0.01)
    assert result["status"] == expected
    assert result["confidence_interval"] is None


def test_drift_is_not_claimed_as_steady_state():
    assert latency_stability([3.] * 10 + [2.] * 10 + [1.] * 10)["status"] == "UNSTABLE"
    assert latency_stability([1.] * 30)["status"] == "STABLE_OBSERVED"
    assert latency_stability([1.] * 3)["status"] == "INSUFFICIENT"


@pytest.mark.parametrize("ram_limit,decision", [(None, "INCONCLUSIVE"), (1, "REJECT")])
def test_real_fresh_process_pair_and_cli_report(tmp_path, capsys, ram_limit, decision):
    pytest.importorskip("onnxruntime")
    report = run_repeated(config(), BenchmarkConfig(1, 2), Tolerances(), "onnx", tmp_path / "paired", trials=1,
                          max_ram_bytes=ram_limit)
    assert len(report["records"]) == 2
    assert report["statistics"]["status"] == decision
    assert all(r["validation"]["passed"] for r in report["records"])
    assert report["records"][0]["source_state_sha256"] == report["records"][1]["source_state_sha256"]
    save_report(report, tmp_path / "paired", tmp_path)
    main(["compare", "--run-dir", str(tmp_path / "paired"), "--json"])
    assert json.loads(capsys.readouterr().out)["experiment"] == "benchmark-repeat"


def test_export_real_reload_and_checksum_rejection(tmp_path):
    report = AutoOptimizer(config(), BenchmarkConfig(1, 2)).run(tmp_path / "run", ())
    destination = tmp_path / "export"
    manifest = export_result(report, destination)
    assert manifest["verified"] and manifest["reload_validation"]["passed"]
    runner = reload_artifact(destination)
    assert runner(torch.ones(1, 16)).shape == (1, 4)
    with (destination / "source-state.pt").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        reload_artifact(destination)
    with pytest.raises(ValueError, match="already exists"):
        export_result(report, destination)


def test_export_changed_factory_is_rejected(tmp_path):
    report = AutoOptimizer(config(), BenchmarkConfig(1, 2)).run(tmp_path / "run", ())
    report["model"]["source_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Source weights"):
        export_result(report, tmp_path / "export")


@pytest.mark.parametrize("phase", ["export", "reload"])
def test_export_rejects_changed_forward_with_identical_weights(tmp_path, monkeypatch, phase):
    import importlib
    import sys
    name = "kernelixx_source_change_fixture"
    path = tmp_path / (name + ".py")
    source = ("from torch import nn\n"
              "class Model(nn.Module):\n"
              "    def __init__(self):\n"
              "        super().__init__()\n"
              "        self.layer = nn.Linear(16, 4)\n"
              "    def forward(self, x):\n"
              "        return self.layer(x)\n"
              "def load_model():\n"
              "    return Model()\n")
    path.write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    settings = ExperimentConfig(model=name + ":load_model", device="cpu", input_shape=(2, 16), validation_batches=1)
    try:
        report = AutoOptimizer(settings, BenchmarkConfig(1, 2)).run(tmp_path / "run", ())
        if phase == "reload":
            assert export_result(report, tmp_path / "export")["verified"]
        path.write_text(source.replace("return self.layer(x)", "return self.layer(x) + 1.0"), encoding="utf-8")
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        with pytest.raises(ValueError, match="Source code changed"):
            if phase == "export":
                export_result(report, tmp_path / "export")
            else:
                reload_artifact(tmp_path / "export")
    finally:
        sys.modules.pop(name, None)


def test_legacy_report_keeps_explicit_unverified_source_status(tmp_path):
    report = AutoOptimizer(config(), BenchmarkConfig(1, 2)).run(tmp_path / "run", ())
    report["model"].pop("source_code_fingerprint")
    result = export_result(report, tmp_path / "export")
    assert result["verified"]
    assert result["source_code_validation"]["status"] == "NOT VERIFIED"
    assert "Legacy" in result["source_code_validation"]["reason"]


def test_real_onnx_export_selected_artifact_reload(tmp_path):
    pytest.importorskip("onnxruntime")
    report = AutoOptimizer(config(), BenchmarkConfig(1, 2)).run(tmp_path / "run", ("onnx",))
    assert report["candidates"][0]["validation"]["passed"]
    # Exercise persistence for an explicitly selected valid ONNX fixture. This
    # is a functional export test, not evidence of timing-based acceptance.
    report["selection"]["winner"] = "onnx"
    manifest = export_result(report, tmp_path / "onnx-export")
    assert manifest["format"] == "onnx" and manifest["verified"]
    assert reload_artifact(tmp_path / "onnx-export")(torch.ones(3, 16)).shape == (3, 4)


@pytest.mark.parametrize("field", ["timeout_seconds", "budget_seconds"])
def test_nonfinite_repeat_budget_is_rejected(tmp_path, field):
    with pytest.raises(ValueError, match="positive"):
        run_repeated(config(), BenchmarkConfig(1, 2), Tolerances(), "onnx", tmp_path, **{field: float("nan")})


@pytest.mark.parametrize("command", ["profile-memory", "optimize-pipeline", "benchmark-repeat", "optimize-llm", "export"])
def test_new_commands_have_help(command, capsys):
    with pytest.raises(SystemExit) as completed:
        parser().parse_args([command, "--help"])
    assert completed.value.code == 0
    assert "--help" in capsys.readouterr().out


def test_gguf_gpu_capability_and_parameter_detection():
    def init(self, *, n_ctx, n_batch, n_threads, n_threads_batch, n_gpu_layers, offload_kqv, seed, verbose):
        pass
    backend = SimpleNamespace(Llama=type("Backend", (), {"__init__": init}), llama_supports_gpu_offload=lambda: False)
    with pytest.raises(ValueError, match="no GPU"):
        runtime_parameters(GGUFConfig("x.gguf"), 1, 5, backend)
    with pytest.raises(ValueError, match="n_ubatch"):
        runtime_parameters(GGUFConfig("x.gguf", n_ubatch=32), 1, 0, backend)
    with pytest.raises(ValueError, match="type_k"):
        runtime_parameters(GGUFConfig("x.gguf", type_k="q8_0"), 1, 0, backend)


def test_generation_honors_native_end_of_turn_tokens(monkeypatch):
    import sys
    from kernelixx.llm.gguf import measure_request
    def generate(*args, **kwargs):
        yield from [11, 99, 12]
    model = SimpleNamespace(reset=lambda: None, set_seed=lambda seed: None, token_eos=lambda: 2,
                            generate=generate, detokenize=lambda tokens: b"output",
                            _model=SimpleNamespace(vocab=object()))
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(llama_vocab_is_eog=lambda vocab, token: token == 99))
    result = measure_request(model, [8, 9], 64, 42)
    assert result["generated_token_ids"] == [11]
    assert result["finish_reason"] == "eog"


def test_new_cli_commands_run_and_validate_config(tmp_path, capsys):
    common = ["--model", "examples.custom_model:load_model", "--device", "cpu", "--input-shape", "2,16",
              "--warmup", "1", "--repeats", "2", "--output", str(tmp_path / "runs"), "--json"]
    main(["profile-memory", *common])
    report = json.loads(capsys.readouterr().out)
    assert report["baseline"]["sampled_peak_ram_bytes"] > 0
    main(["export", "--run-dir", str(tmp_path / "runs" / "latest"), "--output", str(tmp_path / "export"), "--json"])
    assert json.loads(capsys.readouterr().out)["verified"]
    main(["optimize-pipeline", *common, "--pipelines", "fp16", "--max-candidates", "1", "--trials", "1"])
    report = json.loads(capsys.readouterr().out)
    assert report["candidates"][0]["status_detail"] == "SKIPPED — INCOMPATIBLE"
    assert report["selection"]["winner"] == "baseline"
    with pytest.raises(SystemExit) as error:
        main(["benchmark-repeat", *common, "--pipeline", "onnx", "--trials", "0"])
    assert error.value.code == 2


def test_fortran_dataset_order_is_preserved(tmp_path):
    x = np.asfortranarray(np.arange(80, dtype=np.float32).reshape(5, 16))
    np.save(tmp_path / "inputs.npy", x)
    np.save(tmp_path / "labels.npy", np.arange(5, dtype=np.int64))
    with BatchSource(config(dataset=str(tmp_path))) as source:
        assert np.array_equal(torch.cat([b for b, _ in source]).numpy(), x)


@pytest.mark.skipif(not os.environ.get("KERNELIXX_TEST_GGUF_MODEL"), reason="Set KERNELIXX_TEST_GGUF_MODEL for native integration")
def test_real_gguf_template_logits_and_export(tmp_path):
    from kernelixx.llm.gguf import run_gguf
    report = run_gguf(GGUFConfig(os.environ["KERNELIXX_TEST_GGUF_MODEL"], threads=(1, 2), context=128,
                                max_tokens=8, repeats=2, n_batch=64, n_ubatch=32,
                                prompt_format="auto", validate_logits=True), tmp_path / "native")
    assert all(run["status"] == "ok" for run in report["runs"])
    assert all(run["validation"]["passed"] for run in report["runs"])
    assert report["runs"][1]["validation"]["prompt_logit_validation"]["status"] == "PASS"
    assert report["runs"][0]["metrics"]["prompt_processing_tokens_per_second"] > 0
    assert report["runs"][0]["kv_cache"]["exact_cache_bytes"] is None
    # llama.cpp may pad context allocation beyond the requested capacity.
    assert report["config"]["context"] == 128
    assert report["runs"][0]["effective_runtime_parameters"]["n_ctx"] >= 128
    assert report["runs"][0]["effective_runtime_parameters"]["n_batch"] == 64
    assert export_result(report, tmp_path / "gguf-export")["verified"]
    assert reload_artifact(tmp_path / "gguf-export")["runs"][0]["validation"]["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_memory_sources_and_release():
    device = torch.device("cuda:0")
    before = torch.cuda.memory_allocated(device)
    with MemoryMonitor(device) as memory:
        x = torch.ones(256, 256, device=device)
        torch.cuda.synchronize()
    assert memory.metrics["torch_peak_allocated_bytes"] >= before + x.numel() * x.element_size()
    assert memory.metrics["torch_peak_reserved_bytes"] >= memory.metrics["torch_peak_allocated_bytes"]
    del x
    assert torch.cuda.memory_allocated(device) == before

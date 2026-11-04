import copy
import json
import struct
import subprocess

import pytest

from kernelixx.cli import main
from kernelixx.llm.gguf import GGUFConfig, aggregate, file_info, measure_request, prompts, run_gguf, validate_runs
from kernelixx.reporting.reporter import load_report


@pytest.mark.parametrize("kwargs", [
    {"threads": ()}, {"threads": (1, 1)}, {"threads": (0,)}, {"context": 12},
    {"max_tokens": 512}, {"warmup": 0}, {"repeats": 1}, {"seed": -1}, {"prompt_format": "unknown"},
])
def test_gguf_rejects_invalid_workloads(kwargs):
    with pytest.raises(ValueError):
        GGUFConfig("model.gguf", **kwargs)


def test_gguf_checks_header_hash_and_prompt_contract(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, 5))
    info = file_info(path)
    assert (info["gguf_version"], info["tensor_count"], info["metadata_entry_count"]) == (3, 4, 5)
    assert len(info["sha256"]) == 64
    path.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="Truncated"):
        file_info(path)
    path.write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(ValueError, match="magic"):
        file_info(path)
    records = prompts(GGUFConfig("model.gguf"))
    assert len(records) == 3
    assert all(item["rendered"].endswith("<|im_start|>assistant\n") for item in records)
    source = tmp_path / "prompts.json"
    source.write_text(json.dumps([{"id": "one", "text": "hello"}] * 2))
    with pytest.raises(ValueError, match="unique"):
        prompts(GGUFConfig("model.gguf", prompt_file=str(source)))
    source.write_text(json.dumps([{"id": "one", "text": "<|im_start|>user"}]))
    with pytest.raises(ValueError, match="role-control"):
        prompts(GGUFConfig("model.gguf", prompt_file=str(source)))
    assert prompts(GGUFConfig("model.gguf", prompt_file=str(source), prompt_format="raw"))[0]["rendered"] == "<|im_start|>user"


class TokenProbe:
    def __init__(self, tokens):
        self.tokens = tokens
        self.resets = 0

    def reset(self):
        self.resets += 1

    def set_seed(self, seed):
        assert seed == 42

    def token_eos(self):
        return 2

    def generate(self, tokens, **kwargs):
        assert kwargs["reset"] is True and kwargs["temp"] == 0
        yield from self.tokens

    def detokenize(self, tokens):
        return b"output" if tokens else b""


def test_gguf_measures_first_token_and_n_minus_one_decode_intervals(monkeypatch):
    ticks = iter([100, 200, 400, 700])
    monkeypatch.setattr("kernelixx.llm.gguf.time.perf_counter_ns", lambda: next(ticks))
    model = TokenProbe([11, 12, 2])
    result = measure_request(model, [8, 9], 64, 42)
    assert model.resets == 1
    assert result["generated_token_ids"] == [11, 12]
    assert result["finish_reason"] == "eos"
    assert result["time_to_first_token_ms"] == pytest.approx(0.0001)
    assert result["request_latency_ms"] == pytest.approx(0.0006)
    assert result["decode_seconds"] == pytest.approx(0.0000002)
    metrics = aggregate([result])
    assert metrics["decode_tokens_per_second"] == pytest.approx(5_000_000)
    # EOS evaluation is in total request time, but is not counted as answer text.
    assert metrics["generated_tokens"] == 2


@pytest.mark.parametrize("tokens,maximum,reason", [([2], 64, "eos"), ([10, 11, 12], 2, "length")])
def test_gguf_empty_outputs_and_length_limits(monkeypatch, tokens, maximum, reason):
    ticks = iter(range(0, 1000, 100))
    monkeypatch.setattr("kernelixx.llm.gguf.time.perf_counter_ns", lambda: next(ticks))
    result = measure_request(TokenProbe(tokens), [5], maximum, 42)
    assert result["finish_reason"] == reason
    if reason == "eos":
        assert result["generated_tokens"] == 0
        assert result["time_to_first_token_ms"] is None
        assert aggregate([result])["decode_tokens_per_second"] is None
    else:
        assert result["generated_token_ids"] == [10, 11]


def sample():
    return {"prompt_id": "test", "repetition": 1, "prompt_token_ids": [1], "generated_token_ids": [4, 5],
            "prompt_tokens": 1, "generated_tokens": 2, "finish_reason": "eos", "text": "test output",
            "request_latency_ms": 100., "time_to_first_token_ms": 25.,
            "decode_seconds": 0.05, "decode_intervals": 1}


def study():
    run = {"threads": 1, "status": "ok", "samples": [sample()], "metrics": aggregate([sample()])}
    other = copy.deepcopy(run)
    other["threads"] = 4
    other["metrics"]["request_latency_median_ms"] = 50.
    return [run, other]


def test_gguf_withholds_speedup_when_greedy_tokens_differ_or_baseline_varies():
    runs = study()
    validate_runs(runs)
    assert runs[1]["comparison"]["request_latency_speedup"] == 2
    runs = study()
    runs[1]["samples"][0]["generated_token_ids"] = [7, 8]
    validate_runs(runs)
    assert runs[1]["validation"]["passed"] is False
    assert "comparison" not in runs[1]
    runs = study()
    repeated = {**sample(), "repetition": 2, "generated_token_ids": [99]}
    runs[0]["samples"].append(repeated)
    validate_runs(runs)
    assert not runs[0]["validation"]["passed"]
    assert all("comparison" not in run for run in runs)


def test_gguf_worker_failure_and_missing_dependency_are_explicit(tmp_path, monkeypatch):
    config = GGUFConfig(str(tmp_path / "model.gguf"), threads=(1,))
    monkeypatch.setattr("kernelixx.llm.gguf.importlib.util.find_spec", lambda *args: None)
    with pytest.raises(RuntimeError, match="dependencies missing"):
        run_gguf(config, tmp_path)
    monkeypatch.setattr("kernelixx.llm.gguf.importlib.util.find_spec", lambda *args: object())
    (tmp_path / "model.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    monkeypatch.setattr("kernelixx.llm.gguf.subprocess.run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "unsupported model"))
    report = run_gguf(config, tmp_path)
    assert report["runs"][0]["status"] == "failed"
    assert (tmp_path / "threads-1" / "stderr.txt").read_text() == "unsupported model"


def test_gguf_cli_and_existing_compare_roundtrip_reports(tmp_path, monkeypatch, capsys):
    runs = study()
    validate_runs(runs)
    report = {"schema_version": 1, "experiment": "gguf", "model": {"filename": "test.gguf", "file_bytes": 32},
              "hardware": {"device": "cpu"}, "config": {}, "prompts": [{"id": "test", "text": "test"}],
              "runs": runs, "limitations": ["Test probe, not real benchmark evidence."]}
    monkeypatch.setattr("kernelixx.llm.gguf.run_gguf", lambda *args: copy.deepcopy(report))
    root = tmp_path / "runs"
    main(["gguf-benchmark", "--model-file", str(tmp_path / "test.gguf"), "--output", str(root), "--json"])
    assert json.loads(capsys.readouterr().out)["experiment"] == "gguf"
    main(["compare", "--run-dir", str(root / "latest"), "--json"])
    assert json.loads(capsys.readouterr().out)["runs"][1]["validation"]["passed"] is True
    assert "time_to_first_token_median_ms" in (root / "latest" / "metrics.csv").read_text()
    assert b"\r\r\n" not in (root / "latest" / "metrics.csv").read_bytes()
    assert "GGUF CPU experiment" in (root / "latest" / "report.md").read_text()
    assert load_report(root / "latest")["experiment"] == "gguf"
    report["runs"][1] = {"threads": 4, "status": "failed", "reason": "runtime unavailable"}
    with pytest.raises(SystemExit) as failed:
        main(["gguf-benchmark", "--model-file", "test.gguf", "--output", str(tmp_path / "fail")])
    assert failed.value.code == 2

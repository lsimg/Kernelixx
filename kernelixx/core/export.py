"""Execution recipes and checksummed artifacts, accepted only after actual reload."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from kernelixx.core.benchmark import TorchRunner
from kernelixx.core.data import BatchSource, dataset_sha256
from kernelixx.core.optimizer import ExperimentConfig, build_candidate
from kernelixx.core.pipeline import check_compatibility
from kernelixx.core.validator import Tolerances, ValidationAccumulator, run_output
from kernelixx.models.loader import load_model, state_sha256
from kernelixx.optimizations.graph import OnnxRunner
from kernelixx.utils.device import cleanup, resolve_device, seed_everything, system_info


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_source_code(expected: dict[str, Any] | None, actual: dict[str, Any]) -> dict[str, str]:
    if expected is None:
        return {"status": "NOT VERIFIED", "reason": "Legacy report/manifest has no source-code fingerprint"}
    if expected["sha256"] != actual["sha256"]:
        raise ValueError("Source code changed since the recorded experiment/artifact; rerun validation and benchmarking")
    return {"status": "VERIFIED" if actual["complete"] else "NOT VERIFIED",
            "scope": actual["scope"]}


def reload_artifact(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["checksums"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or checksum(path) != expected:
            raise ValueError(f"Artifact checksum/path mismatch: {name}")
    if manifest["format"] == "gguf-runtime-configuration":
        from kernelixx.llm.gguf import GGUFConfig, file_info, run_gguf, validate_runs
        from kernelixx.reporting.reporter import create_run_directory

        config = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
        config["threads"] = tuple(config["threads"])
        config["n_gpu_layers"] = tuple(config["n_gpu_layers"])
        config["prompt_file"] = str((directory / "prompts.json").resolve())
        if file_info(Path(config["model_file"]))["sha256"] != manifest["source_model"]["sha256"]:
            raise ValueError("GGUF source weights checksum mismatch")
        result = run_gguf(GGUFConfig(**config), create_run_directory(directory / "reloads"))
        if any(run["status"] != "ok" for run in result["runs"]):
            raise ValueError("GGUF artifact runtime reload failed; inspect reloads worker logs")
        pair = [copy.deepcopy(manifest["source_report"]["runs"][0]), result["runs"][0]]
        validate_runs(pair)
        if not pair[1]["validation"]["passed"]:
            raise ValueError("GGUF artifact output changed on reload")
        return result
    config = ExperimentConfig(**{**manifest["config"], "input_shape": tuple(manifest["config"]["input_shape"])})
    device, _ = resolve_device(manifest["device"])
    if manifest["device"] == "cuda" and device.type != "cuda":
        raise RuntimeError("This execution recipe requires CUDA")
    if manifest["format"] == "onnx":
        runner = OnnxRunner(directory / manifest["model_file"], config.threads)
        runner.precision = manifest["precision"]
        return runner
    seed_everything(config.seed, config.threads)
    model, info = load_model(config.model, "none")
    verify_source_code(manifest.get("source_code_fingerprint"), info["source_code_fingerprint"])
    model.load_state_dict(torch.load(directory / "source-state.pt", map_location="cpu", weights_only=True))
    if state_sha256(model) != manifest["source_state_sha256"]:
        raise ValueError("Reloaded source state differs")
    method = manifest["method"]
    if method == "baseline":
        return TorchRunner(model, device)
    check_compatibility(method, model, device)
    with BatchSource(config) as source:
        first, _ = next(iter(source))
    return build_candidate(method, model, first, device, directory / "runtime", config.threads)[0]


def export_result(report: dict[str, Any], directory: Path) -> dict[str, Any]:
    if report.get("experiment") == "gguf":
        return export_gguf_result(report, directory)
    if report.get("experiment") != "inference":
        raise ValueError("Export currently requires an inference report; GGUF configs remain in GGUF report JSON")
    if directory.exists():
        raise ValueError("Export destination already exists; choose a new directory")
    config = ExperimentConfig(**{**report["config"], "input_shape": tuple(report["config"]["input_shape"])})
    seed_everything(config.seed, config.threads)
    device, _ = resolve_device(config.device)
    if str(device) != report["baseline"]["device"]:
        raise ValueError("Export verification requires the original execution device")
    if config.dataset and dataset_sha256(config.dataset) != report.get("dataset_sha256"):
        raise ValueError("Validation dataset changed since the report")
    model, info = load_model(config.model, config.weights)
    source_validation = verify_source_code(report["model"].get("source_code_fingerprint"), info["source_code_fingerprint"])
    fingerprint = state_sha256(model)
    if fingerprint != report["model"].get("source_state_sha256"):
        raise ValueError("Source weights changed or the report predates source-state fingerprints; rerun the experiment")
    method = report["selection"]["winner"]
    directory.mkdir(parents=True)
    tolerances = Tolerances(report["baseline_validation"]["atol"], report["baseline_validation"]["rtol"])
    manifest = {"schema_version": 1, "method": method, "source_state_sha256": fingerprint,
                "source_code_fingerprint": info["source_code_fingerprint"], "source_code_validation": source_validation,
                "config": asdict(config), "transforms": [] if method == "baseline" else method.split("+"),
                "device": device.type, "hardware": system_info(), "source_report": report,
                "verified": False, "checksums": {}}
    runner = None
    try:
        with BatchSource(config) as source:
            first, _ = next(iter(source))
            if method.startswith("onnx"):
                check_compatibility(method, model, device)
                runner, details = build_candidate(method, copy.deepcopy(model), first, device, directory, config.threads)
                path = Path(details["export_path"])
                manifest.update(format="onnx", model_file=path.name, precision=runner.precision)
                manifest["checksums"][path.name] = checksum(path)
                del runner
                runner = None
            else:
                torch.save(model.state_dict(), directory / "source-state.pt")
                manifest.update(format="pytorch-execution-recipe", model_file="source-state.pt")
                manifest["checksums"]["source-state.pt"] = checksum(directory / "source-state.pt")
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            runner = reload_artifact(directory)
            # CPU reference is computed one batch at a time, outside timing.
            reference = TorchRunner(model, torch.device("cpu"))
            classification = report["validation_task"] == "classification"
            accumulator = ValidationAccumulator(tolerances, classification, config.validation_chunk_elements)
            for batch, labels in source:
                ref, output = run_output(reference, batch), run_output(runner, batch)
                accumulator.update(ref, output, labels)
                del ref, output
            validation = accumulator.result()
            if validation.get("accuracy_measured"):
                validation["passed"] &= validation["accuracy_drop"] <= config.max_accuracy_drop
            if not validation["passed"]:
                raise ValueError("Reloaded artifact failed numerical/quality validation")
            manifest.update(verified=True, reload_validation=validation)
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            return manifest
    except Exception as exc:
        manifest.update(verified=False, failure=f"{type(exc).__name__}: {exc}")
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        raise
    finally:
        runner = None
        cleanup(device)


def export_gguf_result(report: dict[str, Any], directory: Path) -> dict[str, Any]:
    from kernelixx.llm.gguf import GGUFConfig, file_info, run_gguf, validate_runs

    if directory.exists():
        raise ValueError("Export destination already exists; choose a new directory")
    baseline = report["runs"][0]
    if baseline["status"] != "ok" or not baseline.get("validation", {}).get("passed"):
        raise ValueError("GGUF baseline has no successful validation")
    if file_info(Path(report["config"]["model_file"]))["sha256"] != report["model"]["sha256"]:
        raise ValueError("GGUF source weights changed")
    directory.mkdir(parents=True)
    config = {**report["config"], "threads": (baseline["threads"],),
              "n_gpu_layers": (baseline.get("n_gpu_layers", 0),), "warmup": 1, "repeats": 2}
    prompts = [{key: item[key] for key in ("id", "text", "expected_exact") if key in item} for item in report["prompts"]]
    (directory / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    config["prompt_file"] = str((directory / "prompts.json").resolve())
    (directory / "runtime.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {"schema_version": 1, "format": "gguf-runtime-configuration", "verified": False,
                "source_model": report["model"], "source_report": report,
                "checksums": {name: checksum(directory / name) for name in ("runtime.json", "prompts.json")}}
    try:
        reloaded = run_gguf(GGUFConfig(**config), directory / "reload-check")
        if reloaded["runs"][0]["status"] != "ok":
            raise ValueError("GGUF reload failed; see reload-check worker logs")
        pair = [copy.deepcopy(baseline), reloaded["runs"][0]]
        validate_runs(pair)
        if not pair[1]["validation"]["passed"]:
            raise ValueError("Reloaded GGUF tokens/text/logits do not match the source run")
        manifest.update(verified=True, reload_validation=pair[1]["validation"])
    except Exception as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest

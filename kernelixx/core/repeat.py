"""Fresh-process paired trials with alternating order and conservative decisions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kernelixx.core.benchmark import BenchmarkConfig, TorchRunner, benchmark, comparison
from kernelixx.core.data import BatchSource, dataset_sha256
from kernelixx.core.optimizer import ExperimentConfig, build_candidate, objective_score
from kernelixx.core.pipeline import check_compatibility
from kernelixx.core.validator import Tolerances, ValidationAccumulator, run_output
from kernelixx.models.loader import load_model, state_sha256
from kernelixx.utils.device import resolve_device, seed_everything, system_info


def summarize_ratios(ratios: list[float], minimum_improvement: float) -> dict[str, Any]:
    if any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Ratios must be positive finite observations")
    required = 1 / (1 - minimum_improvement)
    if len(ratios) < 3:
        status = "INCONCLUSIVE"
    elif min(ratios) > required:
        status = "ACCEPT"
    elif max(ratios) <= required:
        status = "REJECT"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "pairs": len(ratios), "paired_ratios": ratios,
            "median_speedup": statistics.median(ratios) if ratios else None,
            "minimum_speedup": min(ratios) if ratios else None,
            "maximum_speedup": max(ratios) if ratios else None,
            "sample_standard_deviation": statistics.stdev(ratios) if len(ratios) > 1 else None,
            "required_ratio": required, "confidence_interval": None,
            "policy": "At least 3 complete pairs, every pair exceeds the margin. Descriptive policy, not significance."}


def worker(payload: dict[str, Any], directory: Path) -> dict[str, Any]:
    config = ExperimentConfig(**{**payload["config"], "input_shape": tuple(payload["config"]["input_shape"])})
    timing, tolerances = BenchmarkConfig(**payload["timing"]), Tolerances(**payload["tolerances"])
    seed_everything(config.seed, config.threads)
    device, warnings = resolve_device(config.device)
    if config.device == "cuda" and device.type != "cuda":
        raise RuntimeError("Requested CUDA is unavailable; refusing to relabel CPU trials")
    model, model_info = load_model(config.model, config.weights)
    fingerprint = state_sha256(model)
    method = payload["method"]
    reference_directory = Path(payload["reference_directory"])
    reference_directory.mkdir(parents=True, exist_ok=True)
    with BatchSource(config) as source:
        first, _ = next(iter(source))
        input_hash = hashlib.sha256(first.numpy().tobytes()).hexdigest()
        start = time.perf_counter()
        if method == "baseline":
            runner = TorchRunner(copy.deepcopy(model), device)
        else:
            check_compatibility(method, model, device)
            runner, _ = build_candidate(method, copy.deepcopy(model), first, device, directory, config.threads)
        setup = time.perf_counter() - start
        dtype = torch.float16 if runner.precision == "fp16" else torch.float32
        metrics = benchmark(runner, first.to(device, dtype=dtype), timing)
        accumulator = ValidationAccumulator(tolerances, config.task == "classification" or
                                            (config.task == "auto" and (config.model in {"resnet18", "resnet50"} or config.dataset is not None)),
                                            config.validation_chunk_elements)
        for index, (batch, labels) in enumerate(source):
            output = run_output(runner, batch)
            path = reference_directory / f"{index}.npy"
            if payload.get("create_reference"):
                np.save(path, output.numpy(), allow_pickle=False)
                ref = output
            else:
                ref = torch.from_numpy(np.load(path, allow_pickle=False))
            accumulator.update(ref, output, labels)
            del output, ref
        validation = accumulator.result()
        if validation.get("accuracy_measured"):
            validation["passed"] &= validation["accuracy_drop"] <= config.max_accuracy_drop
    return {"status": "ok", "method": method, "metrics": metrics, "validation": validation, "pid": os.getpid(),
            "hardware": system_info(),
            "source_state_sha256": fingerprint, "first_input_sha256": input_hash,
            "source_code_sha256": model_info["source_code_fingerprint"]["sha256"],
            "dataset_sha256": dataset_sha256(config.dataset) if config.dataset else None,
            "setup_seconds": setup, "warnings": warnings}


def run_repeated(config: ExperimentConfig, timing: BenchmarkConfig, tolerances: Tolerances,
                 method: str, directory: Path, trials: int = 3, timeout_seconds: float = 300,
                 max_ram_bytes: int | None = None, max_vram_bytes: int | None = None,
                 budget_seconds: float = 1800) -> dict[str, Any]:
    if not 1 <= trials <= 20 or not math.isfinite(timeout_seconds) or not math.isfinite(budget_seconds) or timeout_seconds <= 0 or budget_seconds <= 0:
        raise ValueError("trials must be in [1,20]; timeout and budget must be positive")
    if any(v is not None and v <= 0 for v in (max_ram_bytes, max_vram_bytes)):
        raise ValueError("Resource limits must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    common = {"config": asdict(config), "timing": asdict(timing), "tolerances": asdict(tolerances),
              "reference_directory": str((directory / "reference").resolve())}
    records = []
    deadline = time.monotonic() + budget_seconds

    def execute(label: str, selected: str, create_reference: bool = False) -> dict[str, Any]:
        task = directory / label
        task.mkdir()
        payload = {**common, "method": selected, "create_reference": create_reference}
        source = task / "worker.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = subprocess.run([sys.executable, "-m", "kernelixx.core.repeat", "--worker", str(source.resolve())],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=max(0.01, min(timeout_seconds, deadline - time.monotonic())))
        (task / "stdout.json").write_text(completed.stdout, encoding="utf-8")
        (task / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Worker {label} exited {completed.returncode}; see {task / 'stderr.txt'}")
        return json.loads(completed.stdout)

    failure = None
    ratios: list[float] = []
    resource_rejected = False
    quality_rejected = False
    try:
        reference = execute("reference-run", "baseline", True)
        if not reference["validation"]["passed"]:
            raise ValueError("Reference baseline validation failed; cannot compare candidates")
        for trial in range(1, trials + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Search budget exhausted")
            pair = {}
            for position, selected in enumerate(("baseline", method) if trial % 2 else (method, "baseline"), 1):
                record = execute(f"trial-{trial}-{position}", selected)
                record.update(trial=trial, position=position)
                records.append(record)
                (directory / "completed-runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                for key in ("source_state_sha256", "source_code_sha256", "first_input_sha256", "dataset_sha256"):
                    if record[key] != reference[key]:
                        raise ValueError(f"Worker workload changed: {key}")
                if not record["validation"]["passed"]:
                    quality_rejected = selected != "baseline"
                    raise ValueError("Numerical or labeled quality validation failed in an independent trial")
                pair[selected] = record
            base, opt = pair["baseline"]["metrics"], pair[method]["metrics"]
            if not comparison(base, opt)["comparable"]:
                raise ValueError("Independent workers measured different workloads/devices")
            if max_ram_bytes is not None:
                # Lifetime OS high-water marks also cover loading/setup peaks.
                value = opt["memory"].get("os_process_lifetime_peak_rss_bytes") or opt["sampled_peak_ram_bytes"]
                if value is None:
                    raise ValueError("RAM limit cannot be verified without process RSS")
                resource_rejected |= value > max_ram_bytes
            if max_vram_bytes is not None:
                value = opt["sampled_process_vram_peak_bytes"]
                if value is None:
                    raise ValueError("Process VRAM limit cannot be verified; allocator bytes cannot substitute")
                resource_rejected |= value > max_vram_bytes
            if any(m["latency_stability"]["status"] != "STABLE_OBSERVED" for m in (base, opt)):
                raise ValueError("Steady-state is unverified: insufficient timings or >10% within-run drift; increase warmup/repeats")
            score = objective_score(opt, base, config.objective)
            if score is None:
                raise ValueError("Requested objective has no comparable measurement")
            ratios.append(1 / score)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, TimeoutError) as exc:
        failure = str(exc)
    summary = summarize_ratios(ratios, config.min_improvement)
    summary["objective"] = config.objective
    summary["median_objective_ratio"] = summary["median_speedup"]
    if config.objective != "latency":
        for prefix in ("median", "minimum", "maximum"):
            summary[f"{prefix}_objective_ratio"] = summary.pop(f"{prefix}_speedup")
    if failure:
        summary.update(status="INCONCLUSIVE", reason=failure)
    if quality_rejected:
        summary.update(status="REJECT", reason=failure)
    if resource_rejected:
        summary.update(status="REJECT", reason="Candidate exceeds a measured resource limit")
    report = {"schema_version": 1, "experiment": "benchmark-repeat", "config": asdict(config),
              "timing": asdict(timing), "tolerances": asdict(tolerances), "method": method,
              "records": records, "statistics": summary,
              "selection": {"winner": method if summary["status"] == "ACCEPT" else "baseline",
                            "improved": summary["status"] == "ACCEPT", "status": summary["status"]},
              "resource_limits": {"process_rss_peak_bytes": max_ram_bytes, "sampled_process_vram_bytes": max_vram_bytes,
                                  "ram_source": "OS lifetime peak where available, otherwise sampled post-warmup RSS"},
              "order": "Baseline/candidate, candidate/baseline, alternating; a fresh process for every measurement"}
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated benchmark worker")
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(worker(json.loads(args.worker.read_text(encoding="utf-8")), args.worker.parent), allow_nan=False))


if __name__ == "__main__":
    main()

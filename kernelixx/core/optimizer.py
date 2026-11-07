from __future__ import annotations

import copy
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kernelixx.core.analyzer import profile_model
from kernelixx.core.benchmark import BenchmarkConfig, Runner, TorchRunner, benchmark, comparison, measure_cold_start
from kernelixx.core.validator import Tolerances, ValidationAccumulator, run_output
from kernelixx.core.data import BatchSource, dataset_sha256
from kernelixx.core.memory import MemoryMonitor
from kernelixx.core.pipeline import PIPELINES, check_compatibility, compose
from kernelixx.models.loader import load_model, state_sha256
from kernelixx.optimizations import UnsupportedOptimization
from kernelixx.optimizations.fusion import compile_model
from kernelixx.optimizations.graph import export_onnx
from kernelixx.optimizations.precision import fp16
from kernelixx.optimizations.quantization import quantize
from kernelixx.utils.device import cleanup, resolve_device, seed_everything, synchronize, system_info

METHODS = ("fp16", "onnx", "compile", "quantization")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "resnet18"
    weights: str = "none"
    device: str = "auto"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    seed: int = 42
    threads: int = 1
    validation_batches: int = 3
    dataset: str | None = None
    objective: str = "latency"
    min_improvement: float = 0.01
    max_accuracy_drop: float = 0.01
    task: str = "auto"
    validation_chunk_elements: int = 65536

    def __post_init__(self) -> None:
        if len(self.input_shape) < 2 or min(self.input_shape) < 1:
            raise ValueError("input-shape needs positive dimensions including batch and features")
        if self.validation_chunk_elements < 1:
            raise ValueError("validation-chunk-elements must be positive")
        if self.threads < 1 or self.validation_batches < 1 or not 0 <= self.seed < 2**32:
            raise ValueError("threads/validation-batches must be positive; seed must be in [0, 2**32)")
        if self.objective not in {"latency", "throughput", "memory", "balanced"}:
            raise ValueError("Unknown optimization objective")
        if self.task not in {"auto", "classification", "regression"}:
            raise ValueError("Task must be auto, classification, or regression")
        if self.dataset and self.task == "regression":
            raise ValueError("The labeled NPZ contract supports classification only")
        if not 0 <= self.min_improvement < 1 or not 0 <= self.max_accuracy_drop <= 1:
            raise ValueError("min-improvement must be in [0,1); max-accuracy-drop in [0,1]")


def make_batches(config: ExperimentConfig) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    # Compatibility materializer for existing examples; the optimizer uses BatchSource.
    with BatchSource(config) as source:
        pairs = list(source)
    return [x for x, _ in pairs], [y for _, y in pairs] if config.dataset else None


def objective_score(metrics: dict[str, Any], baseline: dict[str, Any], objective: str) -> float | None:
    """A dimensionless lower-is-better score; baseline always scores one."""
    if not comparison(baseline, metrics)["comparable"]:
        return None
    latency = metrics["latency_median_ms"] / baseline["latency_median_ms"]
    if objective == "latency":
        return latency
    if objective == "throughput":
        return baseline["throughput_samples_per_sec"] / metrics["throughput_samples_per_sec"]
    bm, cm = baseline["peak_gpu_memory_bytes"], metrics["peak_gpu_memory_bytes"]
    if not bm or cm is None:
        return None
    memory = cm / bm
    return memory if objective == "memory" else (latency * memory) ** 0.5


def select_best(baseline: dict[str, Any], candidates: list[dict[str, Any]],
                objective: str, min_improvement: float) -> dict[str, Any]:
    if not candidates:
        return {"winner": "baseline", "improved": False, "objective": objective,
                "reason": "Baseline measured; no optimization candidates were requested."}
    if objective_score(baseline, baseline, objective) is None:
        return {"winner": "baseline", "improved": False, "objective": objective,
                "reason": "Peak GPU memory is unavailable on CPU; memory/balanced cannot be evaluated."}
    winner, best = "baseline", 1.0
    for candidate in candidates:
        if candidate["status"] != "ok" or not candidate["validation"]["passed"]:
            continue
        score = objective_score(candidate["metrics"], baseline, objective)
        candidate["objective_score"] = score
        if score is not None and score < best and score < 1 - min_improvement:
            winner, best = candidate["method"], score
    improved = winner != "baseline"
    return {"winner": winner, "improved": improved, "objective": objective,
            "score": best, "required_fractional_improvement": min_improvement,
            "reason": "Measured improvement with passing validation." if improved else
            "No validated candidate exceeded the baseline by the required margin.",
            "caveat": "Single-session measurements; small gains may be timing noise. Repeat independently."}


def build_candidate(method: str, model: nn.Module, sample: torch.Tensor,
                    device: torch.device, directory: Path, threads: int) -> tuple[Runner, dict[str, Any]]:
    if method == "fp16":
        return fp16(model, device), {"scope": "FP16 parameters, buffers and inputs on CUDA"}
    if method == "quantization":
        return quantize(model, device, owned=True)
    if method == "onnx":
        return export_onnx(model, sample, device, directory, threads)
    if method == "compile":
        return compile_model(model, device)
    if method in PIPELINES:
        return compose(method, model, sample, device, directory, threads)
    raise ValueError(f"Unknown method: {method}")


class AutoOptimizer:
    def __init__(self, config: ExperimentConfig, timing: BenchmarkConfig = BenchmarkConfig(),
                 tolerances: Tolerances = Tolerances()):
        self.config, self.timing, self.tolerances = config, timing, tolerances

    def run(self, directory: Path, methods: tuple[str, ...] = METHODS,
            analyze: bool = False, max_search_seconds: float | None = None) -> dict[str, Any]:
        if any(method not in PIPELINES for method in methods):
            raise ValueError("Unknown optimization method")
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        config = self.config
        seed_everything(config.seed, config.threads)
        device, warnings = resolve_device(config.device)
        with MemoryMonitor() as loading:
            model, info = load_model(config.model, config.weights)
        info["source_state_sha256"] = state_sha256(model)
        classification = config.task == "classification" or (
            config.task == "auto" and (config.model in {"resnet18", "resnet50"} or config.dataset is not None))
        result: dict[str, Any] = {"schema_version": 1, "experiment": "inference",
                                  "hardware": system_info(), "config": asdict(config),
                                  "timing": asdict(self.timing), "model": info,
                                  "model_loading_memory": loading.metrics,
                                  "validation_task": "classification" if classification else "numerical-only",
                                  "warnings": warnings, "candidates": [],
                                  "limitations": [
                                      "Synthetic inputs measure output deviation, not real task accuracy." if config.dataset is None else
                                      "Accuracy covers only the supplied preprocessed dataset.",
                                      "RSS is sampled and includes runtime/mapped files; exact CPU inference peak is not measured.",
                                      "CUDA allocated/reserved metrics are not total process VRAM.",
                                      "Only one target model is resident on GPU; references are streamed from temporary disk.",
                                      "Cold-start includes lazy compilation; steady-state excludes setup, transfers and validation.",
                                      "Single-session candidates require independent confirmation before selection."]}
        if config.dataset:
            result["dataset_sha256"] = dataset_sha256(config.dataset)
        with BatchSource(config) as batches, tempfile.TemporaryDirectory(prefix="kernelixx-reference-") as temporary:
            first, _ = next(iter(batches))
            spool = Path(temporary)
            start = time.perf_counter()
            runner = TorchRunner(copy.deepcopy(model), device)
            result["baseline_setup_seconds"] = time.perf_counter() - start
            sample = first.to(device)
            try:
                cold_ms = measure_cold_start(runner, sample)
                accumulator = ValidationAccumulator(self.tolerances, classification, config.validation_chunk_elements)
                for index, (batch, labels) in enumerate(batches):
                    output = run_output(runner, batch)
                    accumulator.update(output, output, labels)
                    np.save(spool / f"{index}.npy", output.numpy(), allow_pickle=False)
                    del output
                result["baseline_validation"] = accumulator.result()
                if not result["baseline_validation"]["passed"]:
                    raise ValueError("Baseline outputs are nonfinite; cannot compare optimizations")
                result["baseline"] = benchmark(runner, sample, self.timing, cold_start_ms=cold_ms)
                if analyze:
                    try:
                        result["analysis"] = profile_model(runner, sample)
                    except Exception as exc:
                        warnings.append(f"Profiler failed: {type(exc).__name__}: {exc}")
            finally:
                runner = sample = None
                cleanup(device)
            for method in dict.fromkeys(methods):
                entry: dict[str, Any] = {"method": method, "transforms": method.split("+")}
                if max_search_seconds is not None and time.perf_counter() - started >= max_search_seconds:
                    result["candidates"].append({**entry, "status": "skipped", "reason": "Search time budget exhausted"})
                    continue
                runner = sample = None
                start = time.perf_counter()
                try:
                    check_compatibility(method, model, device)
                    runner, details = build_candidate(method, copy.deepcopy(model), first,
                                                      device, directory / method, config.threads)
                    dtype = torch.float16 if runner.precision == "fp16" else torch.float32
                    sample = first.to(device, dtype=dtype)
                    entry.update(details=details, setup_seconds=time.perf_counter() - start)
                    # First forward belongs to cold-start, including lazy compilation.
                    cold_ms = measure_cold_start(runner, sample)
                    accumulator = ValidationAccumulator(self.tolerances, classification, config.validation_chunk_elements)
                    for index, (batch, labels) in enumerate(batches):
                        ref = torch.from_numpy(np.load(spool / f"{index}.npy", allow_pickle=False))
                        output = run_output(runner, batch)
                        accumulator.update(ref, output, labels)
                        del ref, output
                    validation = accumulator.result()
                    if validation.get("accuracy_measured"):
                        validation["max_accuracy_drop"] = config.max_accuracy_drop
                        validation["passed"] &= validation["accuracy_drop"] <= config.max_accuracy_drop
                    metrics = benchmark(runner, sample, self.timing, cold_start_ms=cold_ms)
                    entry.update(status="ok" if validation["passed"] else "rejected",
                                 validation=validation, metrics=metrics,
                                 comparison=comparison(result["baseline"], metrics))
                except UnsupportedOptimization as exc:
                    entry.update(status="skipped", status_detail="SKIPPED — INCOMPATIBLE", reason=str(exc))
                except Exception as exc:
                    entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    runner = sample = None
                    cleanup(device)
                result["candidates"].append(entry)
        selection = select_best(result["baseline"], result["candidates"], config.objective, config.min_improvement)
        if selection["improved"]:
            selection.update(provisional_winner=selection["winner"], winner="baseline", improved=False,
                             status="INCONCLUSIVE", reason="Single-session gain requires independent paired confirmation.")
        else:
            selection["status"] = "REJECT" if methods else "BASELINE"
        result["selection"] = selection
        result["end_to_end_runtime_seconds"] = time.perf_counter() - started
        return result


def optimize_pipeline(config: ExperimentConfig, timing: BenchmarkConfig, tolerances: Tolerances,
                      directory: Path, pipelines: tuple[str, ...], max_candidates: int = 6,
                      budget_seconds: float = 1800, trials: int = 3, timeout_seconds: float = 300,
                      max_ram_bytes: int | None = None, max_vram_bytes: int | None = None) -> dict[str, Any]:
    from kernelixx.core.repeat import run_repeated

    if not 1 <= max_candidates <= len(PIPELINES) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("max-candidates must be in [1,6]; budget must be positive")
    if not 1 <= trials <= 20 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("trials must be in [1,20]; timeout must be positive")
    if any(v is not None and v <= 0 for v in (max_ram_bytes, max_vram_bytes)):
        raise ValueError("Resource limits must be positive")
    chosen = tuple(dict.fromkeys(pipelines))[:max_candidates]
    start = time.perf_counter()
    report = AutoOptimizer(config, timing, tolerances).run(directory, chosen, max_search_seconds=budget_seconds)
    winner, best = "baseline", 1.0
    for entry in report["candidates"]:
        if entry["status"] != "ok":
            continue
        remaining = budget_seconds - (time.perf_counter() - start)
        if remaining <= 0:
            entry["decision"] = "INCONCLUSIVE"
            entry["reason"] = "Search time budget exhausted before repeatability confirmation"
            continue
        repeated = run_repeated(config, timing, tolerances, entry["method"], directory / (entry["method"] + "-repeats"),
                                trials, timeout_seconds, max_ram_bytes, max_vram_bytes, remaining)
        entry["repeatability"] = repeated["statistics"]
        entry["decision"] = repeated["statistics"]["status"]
        ratio = repeated["statistics"]["median_objective_ratio"]
        if entry["decision"] == "ACCEPT" and ratio > best:
            winner, best = entry["method"], ratio
    report["selection"] = {"winner": winner, "improved": winner != "baseline",
                            "status": "ACCEPT" if winner != "baseline" else "INCONCLUSIVE" if any(
                                e.get("decision") == "INCONCLUSIVE" for e in report["candidates"]) else "REJECT",
                            "objective": config.objective,
                            "reason": "Validated, resource-checked improvement in every independent pair." if winner != "baseline" else
                            "No candidate met numerical, quality, resource and repeatability policies; baseline retained."}
    report["search"] = {"max_candidates": max_candidates, "budget_seconds": budget_seconds,
                         "omitted_by_candidate_limit": list(dict.fromkeys(pipelines))[max_candidates:]}
    report["end_to_end_runtime_seconds"] = time.perf_counter() - start
    return report

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kernelixx.core.analyzer import profile_model
from kernelixx.core.benchmark import BenchmarkConfig, Runner, TorchRunner, benchmark, comparison
from kernelixx.core.validator import Tolerances, outputs, validate
from kernelixx.models.loader import load_model
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

    def __post_init__(self) -> None:
        if len(self.input_shape) < 2 or min(self.input_shape) < 1:
            raise ValueError("input-shape needs positive dimensions including batch and features")
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
    if config.dataset is None:
        generator = torch.Generator().manual_seed(config.seed + 1)
        return [torch.randn(config.input_shape, generator=generator)
                for _ in range(config.validation_batches)], None
    with np.load(config.dataset, allow_pickle=False) as data:
        inputs, targets = data["inputs"], data["labels"]
        if inputs.ndim != len(config.input_shape) or tuple(inputs.shape[1:]) != config.input_shape[1:]:
            raise ValueError("Dataset feature shape must match --input-shape (batch may differ)")
        if inputs.dtype != np.float32 or not np.isfinite(inputs).all():
            raise ValueError("NPZ inputs must be finite preprocessed float32")
        if targets.dtype != np.int64 or targets.shape != (len(inputs),) or not len(inputs):
            raise ValueError("NPZ labels must be int64 [N], with N > 0")
        # Keep the final partial batch for validation; time the first batch consistently.
        size = config.input_shape[0]
        batches = [torch.from_numpy(inputs[i:i + size].copy()) for i in range(0, len(inputs), size)]
        labels = [torch.from_numpy(targets[i:i + size].copy()) for i in range(0, len(inputs), size)]
    return batches, labels


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
        return quantize(model, device)
    if method == "onnx":
        return export_onnx(model, sample, device, directory, threads)
    if method == "compile":
        return compile_model(model, device)
    raise ValueError(f"Unknown method: {method}")


class AutoOptimizer:
    def __init__(self, config: ExperimentConfig, timing: BenchmarkConfig = BenchmarkConfig(),
                 tolerances: Tolerances = Tolerances()):
        self.config, self.timing, self.tolerances = config, timing, tolerances

    def run(self, directory: Path, methods: tuple[str, ...] = METHODS,
            analyze: bool = False) -> dict[str, Any]:
        if any(method not in METHODS for method in methods):
            raise ValueError("Unknown optimization method")
        config = self.config
        seed_everything(config.seed, config.threads)
        device, warnings = resolve_device(config.device)
        model, info = load_model(config.model, config.weights)
        batches, labels = make_batches(config)
        classification = config.task == "classification" or (
            config.task == "auto" and (config.model in {"resnet18", "resnet50"} or labels is not None))
        result: dict[str, Any] = {"schema_version": 1, "experiment": "inference",
                                  "hardware": system_info(), "config": asdict(config),
                                  "timing": asdict(self.timing), "model": info,
                                  "validation_task": "classification" if classification else "numerical-only",
                                  "warnings": warnings, "candidates": [],
                                  "limitations": [
                                      "Synthetic inputs measure performance and output deviation, not real accuracy."
                                      if labels is None else "Accuracy covers only the supplied preprocessed dataset.",
                                      "CPU peak inference memory is not measured; weight bytes are a separate metric.",
                                      "GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.",
                                      "Candidates are measured sequentially with only one target model resident on GPU.",
                                      "Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.",
                                      "Training and inference settings are experimental, not universal optimization guarantees."]}
        if config.dataset:
            digest = hashlib.sha256()
            with open(config.dataset, "rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result["dataset_sha256"] = digest.hexdigest()
        runner = TorchRunner(copy.deepcopy(model), device)
        sample = batches[0].to(device)
        reference = outputs(runner, batches)
        baseline_validation = validate(reference, reference, self.tolerances, labels, classification)
        if not baseline_validation["passed"]:
            raise ValueError("Baseline outputs are nonfinite; cannot compare optimizations")
        result["baseline"] = benchmark(runner, sample, self.timing)
        result["baseline_validation"] = baseline_validation
        if analyze:
            try:
                result["analysis"] = profile_model(runner, sample)
            except Exception as exc:
                warnings.append(f"Profiler failed: {type(exc).__name__}: {exc}")
        del runner, sample
        cleanup(device)
        for method in dict.fromkeys(methods):
            entry: dict[str, Any] = {"method": method}
            runner = None
            sample = None
            start = time.perf_counter()
            try:
                runner, details = build_candidate(method, copy.deepcopy(model), batches[0],
                                                  device, directory / method, config.threads)
                dtype = torch.float16 if runner.precision == "fp16" else torch.float32
                sample = batches[0].to(device, dtype=dtype)
                # Force lazy compilation and fail unsupported operations outside benchmark timing.
                with torch.inference_mode():
                    probe = runner(sample)
                    del probe
                synchronize(device)
                entry.update(details=details, setup_seconds=time.perf_counter() - start)
                candidate_outputs = outputs(runner, batches)
                validation = validate(reference, candidate_outputs, self.tolerances, labels, classification)
                if labels is not None and validation.get("baseline_accuracy") is not None:
                    # Subtract integer counts first: 98/100 - 97/100 can round
                    # above 0.01 and incorrectly reject an allowed one-point drop.
                    drop = (validation["baseline_correct"] - validation["optimized_correct"]) / validation["validation_examples"]
                    validation["accuracy_drop"] = drop
                    validation["max_accuracy_drop"] = config.max_accuracy_drop
                    validation["passed"] &= drop <= config.max_accuracy_drop
                metrics = benchmark(runner, sample, self.timing)
                entry.update(status="ok" if validation["passed"] else "rejected",
                             validation=validation, metrics=metrics,
                             comparison=comparison(result["baseline"], metrics))
                del candidate_outputs
            except UnsupportedOptimization as exc:
                entry.update(status="skipped", reason=str(exc))
            except Exception as exc:
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            finally:
                runner = None
                sample = None
                cleanup(device)
            result["candidates"].append(entry)
        result["selection"] = select_best(result["baseline"], result["candidates"],
                                          config.objective, config.min_improvement)
        return result

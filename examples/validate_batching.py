"""Check FP32 output equivalence when regrouping the same labeled images.

Run separately from timed experiments. This diagnostic does not compare latency
across different batch sizes or validate an FP16 candidate against a different batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from kernelixx.core.benchmark import TorchRunner
from kernelixx.core.optimizer import ExperimentConfig, make_batches
from kernelixx.core.validator import Tolerances, outputs, validate
from kernelixx.models.loader import load_model
from kernelixx.utils.device import resolve_device, seed_everything, system_info


def check_batch_equivalence(config: ExperimentConfig, batch_sizes: tuple[int, int]) -> dict[str, Any]:
    if min(batch_sizes) < 1 or config.dataset is None:
        raise ValueError("Two positive batch sizes and a labeled dataset are required")
    seed_everything(config.seed, config.threads)
    device, warnings = resolve_device(config.device)
    batches, labels = make_batches(config)
    if labels is None:
        raise ValueError("Labeled data is required")
    inputs, targets = torch.cat(batches), torch.cat(labels)
    del batches, labels
    model, info = load_model(config.model, config.weights)
    runner = TorchRunner(model, device)
    reference = outputs(runner, list(inputs.split(batch_sizes[0])))
    candidate = outputs(runner, list(inputs.split(batch_sizes[1])))
    result = validate([torch.cat(reference)], [torch.cat(candidate)], Tolerances(), [targets])
    if result.get("baseline_correct") is not None:
        drop = (result["baseline_correct"] - result["optimized_correct"]) / len(targets)
        result["accuracy_drop"] = drop
        result["max_accuracy_drop"] = config.max_accuracy_drop
        result["passed"] &= drop <= config.max_accuracy_drop
    with Path(config.dataset).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"schema_version": 1, "experiment": "batch-equivalence", "model": info["name"],
            "weights": config.weights, "device": str(device), "precision": "fp32",
            "batch_sizes": list(batch_sizes), "seed": config.seed, "threads": config.threads,
            "dataset": config.dataset, "dataset_sha256": digest, "validation": result,
            "hardware": system_info(), "warnings": warnings,
            "note": "Untimed same-model comparison on identical ordered tensors; includes partial final batches."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs=2, default=(1, 8))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Output already exists; choose a new --output path")
        config = ExperimentConfig(model=args.model, weights="default", device=args.device,
                                  dataset=str(args.dataset), seed=args.seed)
        result = check_batch_equivalence(config, tuple(args.batch_sizes))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result, target, indent=2, allow_nan=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result["validation"], indent=2))
    print(f"Saved diagnostic: {args.output}")
    if not result["validation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

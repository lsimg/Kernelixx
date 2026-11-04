"""Repeat the FP32 batch-1/batch-8 study in fresh processes with alternating order."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def describe(values: list[float]) -> dict[str, float]:
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("At least two positive finite measurements are required")
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values),
            "sample_standard_deviation": statistics.stdev(values)}


def summarize_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, dict[int, float]] = {}
    for record in records:
        trial, batch = record["trial"], record["batch_size"]
        if batch not in {1, 8}:
            raise ValueError("This experiment supports batches 1 and 8")
        group = groups.setdefault(trial, {})
        if batch in group:
            raise ValueError("Duplicate trial/batch measurement")
        group[batch] = record["images_per_second"]
    if len(groups) < 2 or any(set(group) != {1, 8} for group in groups.values()):
        raise ValueError("At least two complete paired trials are required")
    singles = [groups[trial][1] for trial in sorted(groups)]
    eights = [groups[trial][8] for trial in sorted(groups)]
    single_stats, eight_stats = describe(singles), describe(eights)
    ratios = [b / a for a, b in zip(singles, eights)]
    return {"trial_count": len(groups), "batch1_images_per_second": single_stats,
            "batch8_images_per_second": eight_stats, "paired_throughput_ratios": ratios,
            "paired_ratio_statistics": describe(ratios),
            "trials_with_batch8_higher_throughput": sum(ratio > 1 for ratio in ratios),
            "note": "Descriptive statistics of whole-process trials; no confidence interval or significance claim."}


def run_child(arguments: list[str], repository: Path, directory: Path, label: str) -> str:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run([sys.executable, "-m", *arguments], cwd=repository, env=environment,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    (directory / f"{label}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (directory / f"{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"{label} exited with code {completed.returncode}; see {directory} logs")
    return completed.stdout


def print_summary(summary: dict[str, Any]) -> None:
    stats = summarize_pairs(summary["records"])
    print(f"Model: {summary['model']}; device: {summary['device']}")
    print("Trial  Batch 1 images/s  Batch 8 images/s  Throughput ratio")
    for trial, ratio in zip(sorted({r["trial"] for r in summary["records"]}), stats["paired_throughput_ratios"]):
        pair = {r["batch_size"]: r for r in summary["records"] if r["trial"] == trial}
        print(f"{trial:>5}  {pair[1]['images_per_second']:>16.2f}  "
              f"{pair[8]['images_per_second']:>16.2f}  {ratio:>16.3f}x")
    ratio = stats["paired_ratio_statistics"]
    print(f"Median paired throughput ratio: {ratio['median']:.3f}x; "
          f"observed range {ratio['minimum']:.3f}x to {ratio['maximum']:.3f}x")
    print("Different batch sizes; these are throughput ratios, not individual-image latency speedups.")
    print("Descriptive observations; no statistical significance claim.")


def main(argv: list[str] | None = None) -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--read-summary", type=Path, help="Read an existing summary without running experiments")
    args = parser.parse_args(argv)
    if args.read_summary:
        try:
            source = args.read_summary / "summary.json" if args.read_summary.is_dir() else args.read_summary
            summary = json.loads(source.read_text(encoding="utf-8"))
            if summary.get("schema_version") != 1 or summary.get("experiment") != "batching-repeat-study":
                raise ValueError("Expected a batching-repeat-study summary")
            print_summary(summary)
            return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(2, f"Error: {exc}\n")
    if args.trials < 2 or args.warmup < 1 or args.repeats < 2:
        parser.error("trials >= 2, warmup >= 1 and repeats >= 2 required")
    dataset = args.dataset.resolve() if args.dataset else repository / f"results/datasets/imagenet-val-{args.model}-200.npz"
    if not dataset.is_file():
        parser.error(f"Dataset not found: {dataset}")
    directory = args.output.resolve() if args.output else repository / "results" / f"batching-{args.model}" / (
        "repeats-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    try:
        directory.mkdir(parents=True, exist_ok=False)
        print(f"Study directory: {directory}", flush=True)
        quality_path = directory / "batch-equivalence.json"
        run_child(["examples.validate_batching", "--model", args.model, "--dataset", str(dataset),
                   "--device", args.device, "--batch-sizes", "1", "8", "--seed", str(args.seed),
                   "--output", str(quality_path)], repository, directory, "quality")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not quality["validation"]["passed"]:
            raise ValueError("Cross-batch quality validation failed")
        if args.device == "cuda" and not quality["device"].startswith("cuda"):
            raise ValueError("CUDA requested but unavailable; no GPU repeat study was performed")
        print("Cross-batch FP32 quality validation: PASS", flush=True)
        records = []
        for trial in range(1, args.trials + 1):
            order = (1, 8) if trial % 2 else (8, 1)
            for position, batch in enumerate(order, 1):
                output = directory / f"trial-{trial}" / f"batch{batch}"
                stdout = run_child([
                    "kernelixx", "benchmark", "--model", args.model, "--weights", "default",
                    "--device", args.device, "--objective", "throughput", "--threads", "1",
                    "--seed", str(args.seed), "--input-shape", f"{batch},3,224,224",
                    "--dataset", str(dataset), "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats), "--output", str(output), "--json"],
                    repository, directory, f"trial-{trial}-batch{batch}")
                report = json.loads(stdout)
                metrics, validation = report["baseline"], report["baseline_validation"]
                if (report["dataset_sha256"] != quality["dataset_sha256"]
                        or metrics["device"] != quality["device"] or metrics["precision"] != "fp32"
                        or metrics["batch_size"] != batch or not validation["passed"]):
                    raise ValueError("A trial changed dataset/device/precision/batch or failed validation")
                record = {"trial": trial, "position_in_pair": position, "batch_size": batch,
                          "images_per_second": metrics["throughput_samples_per_sec"],
                          "batch_latency_median_ms": metrics["latency_median_ms"],
                          "batch_latency_p95_ms": metrics["latency_p95_ms"],
                          "peak_gpu_memory_bytes": metrics["peak_gpu_memory_bytes"],
                          "correct": validation["baseline_correct"], "total": validation["validation_examples"],
                          "report_path": str(Path(report["run_directory"]) / "report.json")}
                records.append(record)
                (directory / "completed-runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(f"Trial {trial}, batch {batch}: {record['images_per_second']:.2f} images/s; "
                      f"correct {record['correct']}/{record['total']}", flush=True)
        summary = {"schema_version": 1, "experiment": "batching-repeat-study", "model": args.model,
                   "device": quality["device"], "dataset_sha256": quality["dataset_sha256"],
                   "seed": args.seed, "threads": 1, "warmup": args.warmup, "repeats": args.repeats,
                   "trial_order": "1,8; 8,1; alternating thereafter", "records": records,
                   "statistics": summarize_pairs(records), "quality_equivalence": quality["validation"]}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        stats = summary["statistics"]
        lines = ["# Repeated FP32 batching study", "", "| Trial | Batch 1 images/s | Batch 8 images/s | Ratio |",
                 "|---:|---:|---:|---:|"]
        for trial, ratio in enumerate(stats["paired_throughput_ratios"], 1):
            pair = {r["batch_size"]: r for r in records if r["trial"] == trial}
            lines.append(f"| {trial} | {pair[1]['images_per_second']:.4f} | {pair[8]['images_per_second']:.4f} | {ratio:.6f} |")
        lines += ["", "```json", json.dumps(stats, indent=2), "```", "",
                  "Throughput ratios compare different batch sizes; they are not single-image latency speedups.",
                  "Fresh process per configuration; alternating order; all measured calls use the first fixed batch.",
                  "Queueing, file decoding, preprocessing, input transfers and process startup are excluded.",
                  "Accuracy covers only the supplied labeled subset. These descriptive observations do not establish statistical significance.", ""]
        (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print_summary(summary)
        print(f"Summary: {directory / 'summary.md'}", flush=True)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Measure two trusted checkouts in alternating fresh processes, same workload.

Example: python tools/compare_implementations.py --original-root .audit/original
--dataset results/datasets/imagenet-val-resnet50-200.npz --device cuda
--weights default --torch-home results/torch-cache --output results/v2-comparison
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker(args):
    sys.path.insert(0, str(args.implementation_root.resolve()))
    import psutil
    from kernelixx.core.benchmark import BenchmarkConfig
    from kernelixx.core.optimizer import AutoOptimizer, ExperimentConfig

    process = psutil.Process()
    before = peak = process.memory_info().rss
    stop = threading.Event()
    def sample():
        nonlocal peak
        while not stop.wait(0.002):
            peak = max(peak, process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        config = ExperimentConfig(model="resnet50", weights=args.weights, device=args.device,
                                  input_shape=(8, 3, 224, 224), dataset=str(args.dataset.resolve()), threads=1)
        methods = ("fp16",) if args.device == "cuda" else ()
        report = AutoOptimizer(config, BenchmarkConfig(args.warmup, args.repeats)).run(args.output, methods)
    finally:
        stop.set()
        thread.join()
    peak = max(peak, process.memory_info().rss)
    result = {"implementation_root": str(args.implementation_root.resolve()), "pid": os.getpid(),
              "sampled_experiment_rss_peak_bytes": peak, "rss_before_bytes": before,
              "rss_peak_delta_bytes": peak - before, "sampling_interval_seconds": 0.002,
              "end_to_end_seconds": time.perf_counter() - start, "report": report}
    (args.output / "measurement.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--weights", choices=("none", "default"), default="none")
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.implementation_root:
        worker(args)
        return
    if not args.original_root or not 1 <= args.trials <= 10:
        parser.error("--original-root and trials in [1,10] are required")
    roots = {"original": args.original_root.resolve(), "optimized": Path(__file__).resolve().parents[1]}
    records = []
    for trial in range(1, args.trials + 1):
        order = ("original", "optimized") if trial % 2 else ("optimized", "original")
        for position, label in enumerate(order, 1):
            directory = args.output / f"trial-{trial}-{label}"
            command = [sys.executable, str(Path(__file__).resolve()), "--implementation-root", str(roots[label]),
                       "--dataset", str(args.dataset.resolve()), "--device", args.device, "--weights", args.weights,
                       "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                       "--output", str(directory.resolve())]
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            (args.output / f"trial-{trial}-{label}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"{label} failed; see saved stderr")
            record = json.loads((directory / "measurement.json").read_text(encoding="utf-8"))
            record.update(trial=trial, position=position, implementation=label)
            if record["report"]["baseline"]["device"].split(":")[0] != args.device:
                raise ValueError("Requested device was not used")
            records.append(record)
            print(f"Trial {trial} {label}: RSS {record['sampled_experiment_rss_peak_bytes']/2**20:.1f} MiB; "
                  f"baseline median {record['report']['baseline']['latency_median_ms']:.3f} ms", flush=True)
    with args.dataset.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    for record in records:
        if record["report"]["dataset_sha256"] != digest:
            raise ValueError("Dataset changed during measurements")
    summary = {"device": args.device, "model": "resnet50", "weights": args.weights,
               "batch_size": 8, "dataset_sha256": digest, "warmup": args.warmup, "repeats": args.repeats,
               "records": records, "note": "Independent paired process observations; no significance claim."}
    summary["paired_ram_ratios"] = [next(r for r in records if r["trial"] == t and r["implementation"] == "original")["sampled_experiment_rss_peak_bytes"] /
                                    next(r for r in records if r["trial"] == t and r["implementation"] == "optimized")["sampled_experiment_rss_peak_bytes"]
                                    for t in range(1, args.trials + 1)]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("Median experiment-RSS ratio:", statistics.median(summary["paired_ram_ratios"]), flush=True)


if __name__ == "__main__":
    main()

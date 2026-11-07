from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from kernelixx import __version__
from kernelixx.core.benchmark import BenchmarkConfig
from kernelixx.core.optimizer import METHODS, AutoOptimizer, ExperimentConfig, optimize_pipeline
from kernelixx.core.pipeline import PIPELINES
from kernelixx.core.validator import Tolerances
from kernelixx.reporting.reporter import create_run_directory, load_report, rows, save_report
from kernelixx.training.benchmark import training_experiment
from kernelixx.utils.device import system_info


def input_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(value) for value in text.split(","))
        if len(shape) < 2 or min(shape) < 1:
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use positive dimensions, e.g. 1,3,224,224") from exc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kernelixx", description="Measured and validated model optimization. CLI only.")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for command, help_text in (("analyze", "Analyze parameters and PyTorch Profiler bottlenecks"),
                                ("benchmark", "Measure the unmodified model"),
                                ("optimize", "Measure and validate one transformation"),
                                ("auto-optimize", "Compare compatible transformations with the baseline"),
                                ("profile-memory", "Profile loading and inference RSS/CUDA memory"),
                                ("optimize-pipeline", "Bounded compatible chains with independent confirmation"),
                                ("benchmark-repeat", "Alternating fresh-process paired measurements")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--model", default="resnet18", help="resnet18, resnet50, or module:factory")
        sub.add_argument("--weights", choices=("none", "default"), default="none", help="default downloads pretrained torchvision weights")
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        sub.add_argument("--input-shape", type=input_shape, default=(1, 3, 224, 224))
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--threads", type=int, default=1)
        sub.add_argument("--warmup", type=int, default=5)
        sub.add_argument("--repeats", type=int, default=30)
        sub.add_argument("--validation-batches", type=int, default=3)
        sub.add_argument("--validation-chunk-elements", type=int, default=65536)
        sub.add_argument("--dataset", help="NPZ or directory with inputs.npy/labels.npy; all samples streamed")
        sub.add_argument("--task", choices=("auto", "classification", "regression"), default="auto",
                         help="auto infers classification only for built-in ResNets or labeled data")
        sub.add_argument("--atol", type=float, default=1e-3)
        sub.add_argument("--rtol", type=float, default=1e-2)
        sub.add_argument("--max-accuracy-drop", type=float, default=0.01, help="Maximum absolute accuracy loss on supplied labels")
        sub.add_argument("--objective", choices=("latency", "memory", "throughput", "balanced"), default="latency")
        sub.add_argument("--min-improvement", type=float, default=0.01, help="Minimum fractional gain to select a candidate")
        sub.add_argument("--output", type=Path, default=Path("results"))
        sub.add_argument("--json", action="store_true", help="Print report JSON instead of tables")
        if command in {"optimize-pipeline", "benchmark-repeat"}:
            sub.add_argument("--trials", type=int, default=3)
            sub.add_argument("--timeout-seconds", type=float, default=300)
            sub.add_argument("--budget-seconds", type=float, default=1800)
            sub.add_argument("--max-ram-mib", type=float, help="Process RSS cap: OS lifetime peak when available, sampled inference RSS otherwise")
            sub.add_argument("--max-vram-mib", type=float, help="Process VRAM limit; requires available NVML process measurements")
        if command == "optimize-pipeline":
            sub.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
            sub.add_argument("--max-candidates", type=int, default=6)
        if command == "benchmark-repeat":
            sub.add_argument("--pipeline", choices=PIPELINES, required=True)
        if command == "optimize":
            sub.add_argument("--method", choices=METHODS, required=True)
        if command == "auto-optimize":
            sub.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    compare = commands.add_parser("compare", help="Read a saved experiment without rerunning")
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--json", action="store_true")
    hardware = commands.add_parser("system-info", help="Inspect installed hardware and libraries")
    hardware.add_argument("--json", action="store_true")
    training = commands.add_parser("train-benchmark", help="Experimental small CNN FP32 versus AMP training")
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--steps", type=int, default=20)
    training.add_argument("--warmup", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--output", type=Path, default=Path("results"))
    training.add_argument("--json", action="store_true")
    export = commands.add_parser("export", help="Export the selected inference artifact and verify actual reload")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--json", action="store_true")
    gguf = commands.add_parser("gguf-benchmark", aliases=["optimize-llm"], help="Measure local GGUF runtime configurations without a server")
    gguf.add_argument("--model-file", required=True, type=Path)
    gguf.add_argument("--n-gpu-layers", nargs="+", type=int, default=[0])
    gguf.add_argument("--n-batch", type=int)
    gguf.add_argument("--n-ubatch", type=int)
    gguf.add_argument("--type-k", choices=("f16", "f32", "q8_0", "q4_0"), default="f16")
    gguf.add_argument("--type-v", choices=("f16", "f32", "q8_0", "q4_0"), default="f16")
    gguf.add_argument("--flash-attn", action="store_true")
    gguf.add_argument("--validate-logits", action="store_true", help="Compare last prompt-token logits through the installed native API")
    gguf.add_argument("--threads", "--n-threads", nargs="+", type=int, default=[1, 4])
    gguf.add_argument("--context", "--n-ctx", type=int, default=512)
    gguf.add_argument("--max-tokens", type=int, default=64)
    gguf.add_argument("--warmup", type=int, default=1)
    gguf.add_argument("--repeats", type=int, default=5)
    gguf.add_argument("--seed", type=int, default=42)
    gguf.add_argument("--prompt-format", choices=("auto", "chatml", "raw"), default="chatml")
    gguf.add_argument("--prompt-file", type=Path, help="JSON list of {id, text} prompts; built-in English prompts by default")
    gguf.add_argument("--timeout-seconds", type=int, default=600, help="Timeout for each isolated CPU configuration")
    gguf.add_argument("--output", type=Path, default=Path("results/gguf"))
    gguf.add_argument("--json", action="store_true")
    return root


def display(report: dict[str, Any], console: Console) -> None:
    if report["experiment"] == "benchmark-repeat":
        console.print_json(data=report["statistics"])
        console.print(f"Selected: {report['selection']['winner']}. Reports: {report.get('run_directory', '')}", markup=False)
        return
    if report["experiment"] == "gguf":
        from kernelixx.llm.reporting import display as gguf_display

        gguf_display(report, console)
        return
    table = Table(title="Kernelixx - actual measurements")
    for title in ("Method", "Status", "Median ms", "p95 ms", "Samples/s", "Speedup"):
        table.add_column(title, no_wrap=title in {"Method", "Status"})
    memory_rows = []
    for row in rows(report):
        def number(key: str, divisor: float = 1) -> str:
            value = row.get(key)
            return "N/A" if value is None else f"{value / divisor:.3f}"
        table.add_row(str(row["method"]), str(row["status"]), number("latency_median_ms"),
                      number("latency_p95_ms"), number("throughput_samples_per_sec"), number("speedup"))
        memory_rows.append((str(row["method"]), number("peak_gpu_memory_bytes", 2**20),
                            number("weight_bytes", 2**20)))
        if row.get("reason"):
            console.print(f"{row['method']}: {row['reason']}", markup=False)
    console.print(table)
    for method, peak, weights in memory_rows:
        console.print(f"{method}: PyTorch allocated peak {peak} MiB; weight tensors {weights} MiB", markup=False)
    for warning in report.get("warnings", []):
        console.print(f"Warning: {warning}", style="yellow", markup=False)
    if report["experiment"] == "inference":
        model = report["model"]
        console.print(f"Model: {model['name']} ({model['architecture']}); parameters: {model['parameters']:,}; "
                      f"trainable: {model['trainable_parameters']:,}; device: {report['baseline']['device']}; "
                      f"batch size: {report['baseline']['batch_size']}", markup=False)
        console.print("Timing: milliseconds per batch; Samples/s counts individual examples.", markup=False)
        console.print(report["baseline_validation"]["accuracy_note"], markup=False)
        baseline_validation = report["baseline_validation"]
        if baseline_validation["accuracy_measured"]:
            count = baseline_validation.get("baseline_correct")
            count_text = f" ({count}/{baseline_validation['validation_examples']})" if count is not None else ""
            console.print(f"Labeled examples: {baseline_validation['validation_examples']}; "
                          f"baseline accuracy: {baseline_validation['baseline_accuracy']:.2%}{count_text}", markup=False)
        for candidate in report["candidates"]:
            if "validation" in candidate:
                validation = candidate["validation"]
                console.print(f"{candidate['method']} validation: {'PASS' if validation['passed'] else 'FAIL'}; "
                              f"max abs error: {validation.get('maximum_absolute_error')}", markup=False)
                if validation.get("accuracy_measured"):
                    count = validation.get("optimized_correct")
                    count_text = f" ({count}/{validation['validation_examples']})" if count is not None else ""
                    console.print(f"{candidate['method']} accuracy: {validation['optimized_accuracy']:.2%}{count_text}; "
                                  f"accuracy drop: {validation['accuracy_drop'] * 100:.2f} percentage points", markup=False)
        selection = report["selection"]
        console.print(f"Selected: {selection['winner']}. {selection['reason']}", markup=False)
    else:
        console.print(report["conclusion"], markup=False)
        if "comparison" in report:
            comparison = report["comparison"]
            console.print(f"Training speedup: {comparison['speedup']:.3f}x; "
                          f"GPU memory reduction: {comparison['memory_reduction_percent']:.2f}%", markup=False)
    if report.get("analysis"):
        console.print("Top profiler operators (self time):", markup=False)
        metric = report["analysis"]["ranked_by"]
        for event in report["analysis"]["bottlenecks"][:5]:
            console.print(f"  {event['operator']}: {event[metric]:.1f} us", markup=False)
    if "run_directory" in report:
        console.print(f"Reports: {report['run_directory']}", markup=False)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    console = Console()
    try:
        if args.command == "system-info":
            if args.json:
                print(json.dumps(system_info(), indent=2))
            else:
                console.print_json(data=system_info())
            return
        if args.command == "export":
            from kernelixx.core.export import export_result
            manifest = export_result(load_report(args.run_dir), args.output.resolve())
            if args.json:
                print(json.dumps(manifest, indent=2, allow_nan=False))
            else:
                console.print(f"Export verified by reload and execution: {args.output.resolve()}", markup=False)
            return
        if args.command == "compare":
            report = load_report(args.run_dir)
        else:
            directory = create_run_directory(args.output)
            if args.command in {"gguf-benchmark", "optimize-llm"}:
                from kernelixx.llm.gguf import GGUFConfig, run_gguf

                config = GGUFConfig(model_file=str(args.model_file.resolve()), threads=tuple(args.threads),
                                    context=args.context, max_tokens=args.max_tokens, warmup=args.warmup,
                                    repeats=args.repeats, seed=args.seed, prompt_format=args.prompt_format,
                                    prompt_file=str(args.prompt_file.resolve()) if args.prompt_file else None,
                                    timeout_seconds=args.timeout_seconds, n_gpu_layers=tuple(args.n_gpu_layers),
                                    n_batch=args.n_batch, n_ubatch=args.n_ubatch, type_k=args.type_k,
                                    type_v=args.type_v, flash_attn=args.flash_attn, validate_logits=args.validate_logits)
                report = run_gguf(config, directory)
            elif args.command == "train-benchmark":
                report = training_experiment(args.device, args.seed, args.threads,
                                             args.steps, args.warmup, args.batch_size)
            else:
                config = ExperimentConfig(model=args.model, weights=args.weights, device=args.device,
                                          input_shape=args.input_shape, seed=args.seed, threads=args.threads,
                                          validation_batches=args.validation_batches, dataset=args.dataset,
                                          objective=args.objective, min_improvement=args.min_improvement,
                                          max_accuracy_drop=args.max_accuracy_drop, task=args.task,
                                          validation_chunk_elements=args.validation_chunk_elements)
                methods = (args.method,) if args.command == "optimize" else (
                    tuple(args.methods) if args.command == "auto-optimize" else ())
                timing, tolerances = BenchmarkConfig(args.warmup, args.repeats), Tolerances(args.atol, args.rtol)
                if args.command in {"optimize-pipeline", "benchmark-repeat"}:
                    limits = [args.max_ram_mib, args.max_vram_mib]
                    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in limits):
                        raise ValueError("Memory limits must be positive finite MiB")
                    ram, vram = (int(v * 2**20) if v is not None else None for v in limits)
                    if args.command == "benchmark-repeat":
                        from kernelixx.core.repeat import run_repeated
                        report = run_repeated(config, timing, tolerances, args.pipeline, directory,
                                              args.trials, args.timeout_seconds, ram, vram, args.budget_seconds)
                    else:
                        report = optimize_pipeline(config, timing, tolerances, directory, tuple(args.pipelines),
                                                   args.max_candidates, args.budget_seconds, args.trials,
                                                   args.timeout_seconds, ram, vram)
                else:
                    report = AutoOptimizer(config, timing, tolerances).run(directory, methods,
                                                                          analyze=args.command == "analyze")
            save_report(report, directory, args.output)
        if args.json:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            display(report, console)
        # Unsupported/rejected single methods are visible failures for scripting.
        if args.command == "optimize" and report["candidates"][0]["status"] != "ok":
            raise SystemExit(2)
        if args.command == "train-benchmark" and any(run["status"] == "failed" for run in report["runs"]):
            raise SystemExit(2)
        if args.command in {"gguf-benchmark", "optimize-llm"} and any(run["status"] != "ok" for run in report["runs"]):
            raise SystemExit(2)
        if args.command == "benchmark-repeat" and report["statistics"].get("reason") and report["statistics"]["status"] == "INCONCLUSIVE":
            raise SystemExit(2)
    except KeyboardInterrupt:
        console = Console(stderr=True)
        console.print("Interrupted.")
        raise SystemExit(130) from None
    except Exception as exc:
        Console(stderr=True).print(f"Error: {type(exc).__name__}: {exc}", style="red", markup=False)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

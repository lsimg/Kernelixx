from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table


def rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"method": f"threads-{run['threads']}-gpu-{run.get('n_gpu_layers', 0)}", "threads": run["threads"],
             "n_gpu_layers": run.get("n_gpu_layers", 0), "status": run["status"],
             **run.get("metrics", {}), **run.get("comparison", {}),
             "token_validation_passed": run.get("validation", {}).get("passed"),
             "reason": run.get("reason", "")} for run in report["runs"]]


def number(value: Any, divisor: float = 1) -> str:
    return "N/A" if value is None else f"{value / divisor:.3f}"


def markdown(report: dict[str, Any]) -> str:
    title = "# Kernelixx GGUF CPU experiment" if report["hardware"].get("device") == "cpu" else "# Kernelixx GGUF runtime experiment"
    lines = [title, "", "## Model and environment", "", "```json",
             json.dumps({"model": report["model"], "hardware": report["hardware"], "config": report["config"]}, indent=2),
             "```", "", "## Measurements", "",
             "| Threads | Requested GPU layers | Status | Request median ms | Request p95 ms | TTFT median ms | Decode tokens/s | Sampled process RSS MiB | Latency ratio | Token agreement |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows(report):
        values = [str(row["threads"]), str(row["n_gpu_layers"]), row["status"],
                  *[number(row.get(key)) for key in ("request_latency_median_ms", "request_latency_p95_ms",
                                                     "time_to_first_token_median_ms", "decode_tokens_per_second")],
                  number(row.get("sampled_process_rss_peak_bytes"), 2**20),
                  number(row.get("request_latency_speedup")), str(row.get("token_validation_passed"))]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "These are native runtime settings for one already-quantized model; requested GPU layers alone do not establish actual offload.",
              "A ratio above one indicates a lower median request latency on this workload only.",
              "Validation checks greedy token identity on these prompts, not answer correctness.",
              "", "## Inputs", "", "```json", json.dumps(report["prompts"], indent=2), "```"]
    for run in report["runs"]:
        lines += ["", f"## Threads: {run['threads']}", "", "```json",
                  json.dumps({key: value for key, value in run.items() if key != "samples"}, indent=2), "```"]
        for sample in run.get("samples", []):
            if sample["repetition"] == 1:
                lines += ["", f"### Response: {sample['prompt_id']}", "", "```text", sample["text"], "```"]
    lines += ["", "## Limitations", "", *[f"- {note}" for note in report["limitations"]]]
    return "\n".join(lines) + "\n"


def display(report: dict[str, Any], console: Console) -> None:
    table = Table(title="Kernelixx - measured native GGUF execution")
    for title in ("Threads", "GPU layers requested", "Status", "Median ms", "p95 ms", "TTFT ms", "Decode tok/s", "RSS MiB", "Ratio"):
        table.add_column(title)
    for row in rows(report):
        table.add_row(str(row["threads"]), str(row["n_gpu_layers"]), row["status"],
                      *[number(row.get(key)) for key in ("request_latency_median_ms", "request_latency_p95_ms",
                                                         "time_to_first_token_median_ms", "decode_tokens_per_second")],
                      number(row.get("sampled_process_rss_peak_bytes"), 2**20),
                      number(row.get("request_latency_speedup")))
    console.print(table)
    console.print(f"Model: {report['model']['filename']}; GGUF file: {report['model']['file_bytes'] / 2**20:.3f} MiB", markup=False)
    console.print("Already-quantized model; runtime comparison. See explicit task/logit checks in JSON; no quantization-savings claim.")
    for run in report["runs"]:
        validation = run.get("validation", {})
        console.print(f"Threads {run['threads']}: token agreement {validation.get('passed', 'N/A')}; "
                      f"{run.get('reason', 'full outputs in report.json')}", markup=False)
    console.print(f"Reports: {report['run_directory']}", markup=False)

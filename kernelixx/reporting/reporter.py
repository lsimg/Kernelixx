from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_run_directory(root: Path) -> Path:
    directory = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    if report["experiment"] == "training":
        for run in report["runs"]:
            metrics = run.get("metrics", {})
            comparison = report.get("comparison", {}) if run["method"] == "amp-fp16" else {}
            records.append({"method": run["method"], "status": run["status"], **metrics,
                            **comparison,
                            "reason": run.get("reason", "")})
        return records
    records.append({"method": "baseline", "status": "ok", **report["baseline"]})
    for candidate in report["candidates"]:
        validation = candidate.get("validation", {})
        records.append({"method": candidate["method"], "status": candidate["status"],
                        **candidate.get("metrics", {}), **candidate.get("comparison", {}),
                        "validation_passed": validation.get("passed"),
                        "maximum_absolute_error": validation.get("maximum_absolute_error"),
                        "mean_absolute_error": validation.get("mean_absolute_error"),
                        "maximum_relative_error": validation.get("maximum_relative_error"),
                        "class_agreement": validation.get("class_agreement"),
                        "baseline_accuracy": validation.get("baseline_accuracy"),
                        "optimized_accuracy": validation.get("optimized_accuracy"),
                        "reason": candidate.get("reason", "")})
    return records


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Kernelixx measured experiment", "", f"Experiment: {report['experiment']}", "",
             "## Environment", "", "```json", json.dumps(report["hardware"], indent=2), "```", "",
             "## Configuration", "", "```json", json.dumps(report["config"], indent=2), "```", ""]
    if "model" in report:
        info = {k: v for k, v in report["model"].items() if k != "architecture_repr"}
        lines += ["## Model", "", "```json", json.dumps(info, indent=2), "```", ""]
    lines += ["## Measurements", "", "| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows(report):
        values = [row.get(key) for key in ("method", "status", "latency_median_ms", "latency_p95_ms",
                                          "throughput_samples_per_sec", "peak_gpu_memory_bytes", "weight_bytes", "speedup")]
        lines.append("| " + " | ".join("N/A" if value is None else
                                          f"{value:.4f}" if isinstance(value, float) else str(value)
                                          for value in values) + " |")
    if "selection" in report:
        lines += ["", "## Conclusion", "", f"Selected: **{report['selection']['winner']}**. {report['selection']['reason']}",
                  "", report["selection"].get("caveat", "")]
    if "baseline_validation" in report:
        lines += ["", "## Baseline validation", "", "```json",
                  json.dumps(report["baseline_validation"], indent=2), "```"]
    if "analysis" in report:
        lines += ["", "## Profile / potential bottlenecks", "", "```json",
                  json.dumps(report["analysis"], indent=2), "```"]
    for candidate in report.get("candidates", []):
        lines += ["", f"## Candidate: {candidate['method']}", "", "```json",
                  json.dumps({k: v for k, v in candidate.items() if k != "metrics"}, indent=2), "```"]
    if report["experiment"] == "training":
        lines += ["", "## Loss trajectories", "", "```json", json.dumps(report["runs"], indent=2), "```",
                  "", report["conclusion"]]
        if "comparison" in report:
            lines += ["", "## Training comparison", "", "```json", json.dumps(report["comparison"], indent=2), "```"]
    lines += ["", "## Limitations", ""]
    lines += [f"- {note}" for note in report.get("limitations", []) + report.get("warnings", [])]
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], directory: Path, root: Path) -> Path:
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["run_directory"] = str(directory.resolve())
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "report.md").write_text(markdown(report), encoding="utf-8")
    records = rows(report)
    keys = list(dict.fromkeys(key for row in records for key in row
                             if not isinstance(row[key], (dict, list))))
    # newline="" prevents Windows from translating CSV's CRLF into CRCRLF.
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for filename in ("report.json", "report.md", "metrics.csv"):
        shutil.copy2(directory / filename, latest / filename)
    return directory


def load_report(directory: Path) -> dict[str, Any]:
    with (directory / "report.json").open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("schema_version") != 1 or report.get("experiment") not in {"inference", "training"}:
        raise ValueError("Unsupported Kernelixx report schema")
    return report

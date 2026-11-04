"""Compare ResNet18/50 photo reviews, matching images by manifest path AND SHA-256.

Run inspect_predictions for each prepared model dataset first. This CLI reads
their measured answers and writes static reports; it does not launch inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from rich.console import Console
from rich.table import Table
from torchvision import models

from examples.inspect_predictions import checked_manifest, font, sha256

GROUPS = {
    "fixed": "ResNet50 corrected ResNet18's mistake",
    "regressed": "ResNet18 was correct; ResNet50 was wrong",
    "both_wrong": "Both models were wrong",
    "both_correct": "Both models were correct",
}


def read_review(path: Path, expected_model: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review = json.loads(path.read_text(encoding="utf-8"))
    summary, rows = review["summary"], review["predictions"]
    if summary["model"] != expected_model:
        raise ValueError(f"Expected a {expected_model} review: {path}")
    weights = (models.ResNet18_Weights.DEFAULT if expected_model == "resnet18"
               else models.ResNet50_Weights.DEFAULT)
    if summary["weights"] != str(weights) or summary["device"] != "cpu" or summary["batch_size"] != 1:
        raise ValueError("Compare pretrained FP32 CPU batch-1 reviews from inspect_predictions")
    dataset = Path(summary["dataset"])
    metadata = json.loads(Path(str(dataset) + ".json").read_text(encoding="utf-8"))
    if sha256(dataset) != summary["dataset_sha256"] or metadata["npz_sha256"] != summary["dataset_sha256"]:
        raise ValueError("Review dataset no longer matches its recorded hash")
    if (metadata["model"] != expected_model or metadata["weights"] != str(weights)
            or metadata["preprocessing"] != repr(weights.transforms())):
        raise ValueError("Review model and dataset preprocessing differ")
    with np.load(dataset, allow_pickle=False) as data:
        labels = data["labels"].tolist()
    paths = checked_manifest(metadata, labels, Path(summary["image_root"]))
    if len(rows) != len(paths) or summary["examples"] != len(rows) or not rows:
        raise ValueError("Review does not cover every manifest photo")
    categories = weights.meta["categories"]
    enriched = []
    for index, (row, entry, image, label) in enumerate(zip(rows, metadata["manifest"], paths, labels)):
        prediction = row["predicted_index"]
        if (row["index"] != index + 1 or Path(row["image_path"]).resolve() != image
                or row["expected_index"] != label or not 0 <= prediction < len(categories)
                or row["expected"] != categories[label] or row["predicted"] != categories[prediction]
                or row["correct"] is not (prediction == label)):
            raise ValueError("Review photo order, labels or predictions are inconsistent")
        enriched.append({**row, "source_path": entry["path"], "image_sha256": entry["sha256"]})
    count = sum(row["correct"] for row in enriched)
    if summary["correct"] != count or summary["errors"] != len(rows) - count:
        raise ValueError("Review summary does not match individual predictions")
    return summary, enriched


def compare_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match exact images independently of row order; do not treat net gain as fixes."""
    def indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        result = {(row["source_path"], row["image_sha256"]): row for row in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError("Review contains duplicate images or is empty")
        return result

    first, second = indexed(left), indexed(right)
    if first.keys() != second.keys():
        raise ValueError("Reviews must contain exactly the same original photos and SHA-256 hashes")
    result = []
    for photo, a in first.items():
        b = second[photo]
        if a["expected_index"] != b["expected_index"] or a["expected"] != b["expected"]:
            raise ValueError("The same photograph has different supplied labels")
        ac = a["predicted_index"] == a["expected_index"]
        bc = b["predicted_index"] == b["expected_index"]
        outcome = ("both_correct" if bc else "regressed") if ac else ("fixed" if bc else "both_wrong")
        result.append({"index": a["index"], "image_path": a["image_path"],
                       "source_path": a["source_path"], "image_sha256": a["image_sha256"],
                       "expected_index": a["expected_index"], "expected": a["expected"],
                       "resnet18_index": a["predicted_index"], "resnet18": a["predicted"],
                       "resnet18_correct": ac, "resnet50_index": b["predicted_index"],
                       "resnet50": b["predicted"], "resnet50_correct": bc, "outcome": outcome})
    return result


def comparison_sheet(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    columns, width, height = 3, 440, 350
    canvas = Image.new("RGB", (columns * width, 64 + math.ceil(len(rows) / columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 14), title, font=font(22), fill="black")
    for position, row in enumerate(rows):
        x, y = position % columns * width, 64 + position // columns * height
        draw.rectangle((x + 3, y + 3, x + width - 3, y + height - 3), outline="#888888", width=2)
        draw.text((x + 12, y + 9), f"Photo #{row['index']:03d}", font=font(18), fill="black")
        with Image.open(row["image_path"]) as original:
            photo = ImageOps.contain(original.convert("RGB"), (410, 200))
        canvas.paste(photo, (x + (width - photo.width) // 2, y + 36 + (200 - photo.height) // 2))
        offset = 246
        for text, color in (
            (f"Expected: {row['expected']}", "black"),
            (f"ResNet18: {row['resnet18']}", "#166534" if row["resnet18_correct"] else "#b91c1c"),
            (f"ResNet50: {row['resnet50']}", "#166534" if row["resnet50_correct"] else "#b91c1c"),
        ):
            for line in textwrap.wrap(text, 46):
                draw.text((x + 12, y + offset), line, font=font(16), fill=color)
                offset += 19
    canvas.save(path)


def compare_reviews(resnet18: Path, resnet50: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Comparison output exists; choose a new --output directory")
    summary18, rows18 = read_review(resnet18, "resnet18")
    summary50, rows50 = read_review(resnet50, "resnet50")
    rows = compare_rows(rows18, rows50)
    counts = {key: sum(row["outcome"] == key for row in rows) for key in GROUPS}
    total = len(rows)
    correct18, correct50 = counts["both_correct"] + counts["regressed"], counts["both_correct"] + counts["fixed"]
    summary: dict[str, Any] = {
        "examples": total, "resnet18_correct": correct18, "resnet50_correct": correct50,
        "resnet18_accuracy": correct18 / total, "resnet50_accuracy": correct50 / total,
        "accuracy_difference_percentage_points": (correct50 - correct18) * 100 / total,
        "counts": counts, "reviews": [{"path": str(path.resolve()), "sha256": sha256(path)}
                                        for path in (resnet18, resnet50)],
        "source_summaries": [summary18, summary50],
        "note": "Same original photos and labels; model-specific DEFAULT weights and preprocessing differ. "
                "Exact-label accuracy on this small subset only; no causal architecture-depth claim.",
    }
    output.mkdir(parents=True, exist_ok=False)
    sheets = {}
    for key, title in GROUPS.items():
        selected = [row for row in rows if row["outcome"] == key]
        with (output / f"{key}.csv").open("x", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
        sheets[key] = []
        if key == "both_correct":
            continue
        for start in range(0, len(selected), 12):
            name = f"{key}-{start // 12 + 1:02d}.png"
            comparison_sheet(selected[start:start + 12], output / name, title)
            sheets[key].append(name)
    summary["contact_sheets"] = sheets
    with (output / "comparison.json").open("x", encoding="utf-8") as target:
        json.dump({"summary": summary, "predictions": rows}, target, indent=2, ensure_ascii=False, allow_nan=False)
    with (output / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ResNet18 versus ResNet50: the same photos", "",
             f"ResNet18: **{correct18}/{total} ({correct18 / total:.2%})**. "
             f"ResNet50: **{correct50}/{total} ({correct50 / total:.2%})**.", "",
             f"Net gain: **{correct50 - correct18:+d}** correct answers; "
             f"accuracy difference: **{summary['accuracy_difference_percentage_points']:+.2f} percentage points**.", "",
             "Green text = correct answer; red text = disagreement with the supplied label.", "",
             "Each card shows the original photo. The models use their own official preprocessing; "
             "their exact crops are in the separate prediction reviews.", "", summary["note"], ""]
    for key, title in GROUPS.items():
        lines += [f"## {title}: {counts[key]}", ""]
        for name in sheets[key]:
            lines += [f"![{title}]({name})", ""]
    lines += ["## Every photo", "", "| # | Photo | Expected | ResNet18 | ResNet50 | Outcome |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        link = quote(Path(os.path.relpath(row["image_path"], output)).as_posix(), safe="/.")
        lines.append(f"| {row['index']} | [photo]({link}) | {row['expected']} | {row['resnet18']} | "
                     f"{row['resnet50']} | {row['outcome']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console = Console()
    console.print(f"ResNet18: {correct18}/{total}; ResNet50: {correct50}/{total}")
    table = Table(title="Same-photo comparison")
    table.add_column("Outcome")
    table.add_column("Photos", justify="right")
    for key, title in GROUPS.items():
        table.add_row(title, str(counts[key]))
    console.print(table)
    console.print(f"Report: {output.resolve() / 'report.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet18-review", type=Path, required=True, help="ResNet18 predictions.json")
    parser.add_argument("--resnet50-review", type=Path, required=True, help="ResNet50 predictions.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=Path(f"results/model-comparisons/{stamp}-{uuid4().hex[:8]}"))
    args = parser.parse_args()
    try:
        compare_reviews(args.resnet18_review, args.resnet50_review, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

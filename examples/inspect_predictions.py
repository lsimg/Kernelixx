"""Match the prepared Imagenette photos to real ResNet predictions, from the CLI.

Writes CSV/JSON/Markdown and static PNG contact sheets; creates no graphical app.
Uses all 1000 ImageNet outputs, with the same prepared inputs as the benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rich.console import Console
from rich.table import Table
from torchvision import models

from kernelixx.core.optimizer import ExperimentConfig, make_batches
from kernelixx.core.validator import Tolerances, checked_output, validate
from kernelixx.models.loader import load_model
from kernelixx.optimizations.graph import OnnxRunner
from kernelixx.utils.device import seed_everything


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checked_manifest(metadata: dict[str, Any], labels: Sequence[int],
                     image_root: Path) -> list[Path]:
    """Verify ordered labels, original files and hashes before associating photos."""
    manifest = metadata["manifest"]
    if len(manifest) != len(labels) or metadata["samples"] != len(labels):
        raise ValueError("Manifest length does not match the prepared dataset")
    root = image_root.resolve()
    paths = []
    for entry, label in zip(manifest, labels):
        if entry["imagenet_target"] != label:
            raise ValueError("Manifest labels do not match the NPZ label order")
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest image path escapes --image-root")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Original image hash mismatch: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Manifest contains duplicate image paths")
    return paths


def prediction_rows(logits: torch.Tensor, labels: Sequence[int], categories: Sequence[str],
                    paths: Sequence[Path], onnx_logits: torch.Tensor | None = None) -> list[dict[str, Any]]:
    """Keep dataset order; reject invalid shapes, labels and nonfinite scores."""
    logits = checked_output(logits)
    if logits.shape != (len(labels), len(categories)) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite logits [number of photos, number of categories]")
    if len(paths) != len(labels) or any(label < 0 or label >= len(categories) for label in labels):
        raise ValueError("Image paths and valid labels must match the model output")
    predictions = logits.argmax(1).tolist()
    onnx_predictions = None
    if onnx_logits is not None:
        onnx_logits = checked_output(onnx_logits)
        if onnx_logits.shape != logits.shape or not torch.isfinite(onnx_logits).all():
            raise ValueError("ONNX output must match the finite PyTorch output shape")
        onnx_predictions = onnx_logits.argmax(1).tolist()
    rows = []
    for index, (label, prediction, path) in enumerate(zip(labels, predictions, paths)):
        row: dict[str, Any] = {
            "index": index + 1, "image_path": str(path),
            "expected_index": int(label), "expected": categories[label],
            "predicted_index": prediction, "predicted": categories[prediction],
            "correct": prediction == label,
        }
        if onnx_predictions is not None:
            other = onnx_predictions[index]
            row.update(onnx_index=other, onnx_predicted=categories[other],
                       onnx_correct=other == label, onnx_matches_pytorch=other == prediction)
        rows.append(row)
    return rows


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(rows: list[dict[str, Any]], inputs: torch.Tensor, path: Path,
                  title: str) -> None:
    """Show original photos AND the exact resized/cropped pixels seen by ResNet."""
    columns, width, height = 3, 440, 320
    canvas = Image.new("RGB", (columns * width, 64 + math.ceil(len(rows) / columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), title, font=font(24), fill="black")
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    for position, row in enumerate(rows):
        x, y = position % columns * width, 64 + position // columns * height
        color = "#166534" if row["correct"] else "#b91c1c"
        draw.rectangle((x + 3, y + 3, x + width - 3, y + height - 3), outline=color, width=3)
        draw.text((x + 12, y + 8), f"#{row['index']:03d}  {'CORRECT' if row['correct'] else 'WRONG'}",
                  font=font(18), fill=color)
        with Image.open(row["image_path"]) as source:
            original = ImageOps.contain(source.convert("RGB"), (200, 180))
        pixels = inputs[row["index"] - 1].cpu() * std + mean
        array = (pixels.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        crop = ImageOps.contain(Image.fromarray(array), (200, 180))
        canvas.paste(original, (x + 12 + (200 - original.width) // 2, y + 40 + (180 - original.height) // 2))
        canvas.paste(crop, (x + 228 + (200 - crop.width) // 2, y + 40 + (180 - crop.height) // 2))
        draw.text((x + 12, y + 222), "Original photo", font=font(14), fill="black")
        draw.text((x + 228, y + 222), "Model input (crop)", font=font(14), fill="black")
        for line_index, line in enumerate(textwrap.wrap(f"Expected: {row['expected']}", 47)
                                           + textwrap.wrap(f"Predicted: {row['predicted']}", 47)):
            draw.text((x + 12, y + 247 + 19 * line_index), line, font=font(16), fill="black")
    canvas.save(path)


def write_reports(output: Path, summary: dict[str, Any], rows: list[dict[str, Any]],
                  inputs: torch.Tensor) -> None:
    # Preserve every earlier review. All inputs are checked before creating output.
    output.mkdir(parents=True, exist_ok=False)
    errors = [row for row in rows if not row["correct"]]
    sheets = []
    for start in range(0, len(errors), 12):
        name = f"errors-{start // 12 + 1:02d}.png"
        contact_sheet(errors[start:start + 12], inputs, output / name,
                      f"ResNet mistakes: {start + 1}-{min(start + 12, len(errors))} of {len(errors)}")
        sheets.append(name)
    correct = [row for row in rows if row["correct"]][:6]
    if correct:
        contact_sheet(correct, inputs, output / "correct-examples.png", "Examples of correct predictions")
    summary["error_contact_sheets"] = sheets
    with (output / "predictions.json").open("x", encoding="utf-8") as target:
        json.dump({"summary": summary, "predictions": rows}, target, indent=2, ensure_ascii=False, allow_nan=False)
    for name, selected in (("predictions.csv", rows), ("errors.csv", errors)):
        with (output / name).open("x", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
    lines = ["# Photo-by-photo prediction review", "",
             f"Model: {summary['model']}; pretrained weights: {summary['weights']}.", "",
             f"Correct: **{summary['correct']}/{summary['examples']} ({summary['accuracy']:.2%})**; "
             f"mistakes: **{summary['errors']}**.", "",
             "Each card shows the original photo on the left and the exact prepared model crop on the right.",
             "Expected = dataset label; Predicted = model answer. Categories use the pretrained model's English names.", "",
             "The model chooses among all 1000 ImageNet classes. Accuracy covers this small subset only.", ""]
    if "onnx_correct" in summary:
        lines += [f"ONNX correct: **{summary['onnx_correct']}/{summary['examples']}**; "
                  f"predictions different from PyTorch: **{summary['onnx_changed_predictions']}**.", ""]
    lines += ["## Mistakes", ""]
    for name in sheets:
        lines += [f"![Mistakes]({name})", ""]
    if not sheets:
        lines += ["No mistakes on this dataset.", ""]
    lines += ["## Every photo", "", "| # | Photo | Expected | PyTorch | Correct | ONNX |",
              "|---|---|---|---|---|---|"]
    for row in rows:
        relative = quote(Path(os.path.relpath(row["image_path"], output)).as_posix(), safe="/.")
        lines.append(f"| {row['index']} | [photo]({relative}) | {row['expected']} | {row['predicted']} | "
                     f"{'yes' if row['correct'] else 'no'} | {row.get('onnx_predicted', 'not tested')} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_predictions(dataset: Path, image_root: Path, output: Path,
                        onnx_path: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Review output already exists; choose a new --output directory")
    metadata = json.loads(Path(str(dataset) + ".json").read_text(encoding="utf-8"))
    name = metadata["model"]
    if name not in {"resnet18", "resnet50"}:
        raise ValueError("Review supports the prepared ResNet18/ResNet50 datasets")
    weights = (models.ResNet18_Weights.DEFAULT if name == "resnet18" else models.ResNet50_Weights.DEFAULT)
    if metadata["weights"] != str(weights) or metadata["preprocessing"] != repr(weights.transforms()):
        raise ValueError("Dataset preprocessing/weights differ from this pretrained model")
    if sha256(dataset) != metadata["npz_sha256"]:
        raise ValueError("Prepared dataset hash differs from its manifest")
    seed_everything(metadata["seed"], threads=1)
    batches, label_batches = make_batches(ExperimentConfig(model=name, weights="default", device="cpu",
                                                          dataset=str(dataset), input_shape=(1, 3, 224, 224)))
    if label_batches is None:
        raise ValueError("Photo review requires dataset labels")
    labels = torch.cat(label_batches).tolist()
    paths = checked_manifest(metadata, labels, image_root)
    model, _ = load_model(name, "default")
    reference, optimized = [], []
    onnx = OnnxRunner(onnx_path, threads=1) if onnx_path is not None else None
    console = Console()
    with torch.inference_mode(), console.status(f"Checking {len(labels)} photos on CPU..."):
        for batch in batches:
            reference.append(checked_output(model(batch)))
            if onnx is not None:
                optimized.append(checked_output(onnx(batch)))
    logits = torch.cat(reference)
    rows = prediction_rows(logits, labels, weights.meta["categories"], paths,
                           torch.cat(optimized) if optimized else None)
    count = sum(row["correct"] for row in rows)
    summary: dict[str, Any] = {
        "model": name, "weights": str(weights), "device": "cpu", "batch_size": 1,
        "seed": metadata["seed"], "torch_version": str(torch.__version__),
        "dataset": str(dataset.resolve()), "dataset_sha256": metadata["npz_sha256"],
        "image_root": str(image_root.resolve()), "examples": len(rows), "correct": count,
        "errors": len(rows) - count, "accuracy": count / len(rows),
        "note": "All prepared photos, in manifest order; all 1000 classes. No new training or timing experiment.",
    }
    if optimized:
        summary.update(onnx_path=str(onnx_path.resolve()), onnx_sha256=sha256(onnx_path),
                       onnx_correct=sum(row["onnx_correct"] for row in rows),
                       onnx_changed_predictions=sum(not row["onnx_matches_pytorch"] for row in rows),
                       onnx_validation=validate(reference, optimized, Tolerances(), label_batches))
    write_reports(output, summary, rows, torch.cat(batches))
    console.print(f"Correct: {count}/{len(rows)} ({count / len(rows):.2%}); mistakes: {len(rows) - count}")
    if optimized:
        console.print(f"ONNX correct: {summary['onnx_correct']}/{len(rows)}; "
                      f"changed predictions: {summary['onnx_changed_predictions']}")
    table = Table(title="First 10 mistakes (all mistakes are saved in errors.csv)")
    for title in ("Photo #", "Expected", "Model answer"):
        table.add_column(title)
    for row in [row for row in rows if not row["correct"]][:10]:
        table.add_row(str(row["index"]), row["expected"], row["predicted"])
    console.print(table)
    console.print(f"Review: {output.resolve() / 'report.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("results/datasets/imagenet-val-resnet18-200.npz"))
    parser.add_argument("--image-root", type=Path, default=Path("results/datasets/imagenette/imagenette2-160"))
    parser.add_argument("--onnx", type=Path, help="Optional saved ONNX model to check on the same photos")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=Path(f"results/predictions/{stamp}-{uuid4().hex[:8]}"))
    args = parser.parse_args()
    try:
        inspect_predictions(args.dataset, args.image_root, args.output, args.onnx)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

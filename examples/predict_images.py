"""Run pretrained ResNet18 and ResNet50 on user-selected images from the CLI.

Writes real top-5 outputs and a static comparison image. No training or accuracy
claim is made without exact class labels. Original images are read, never modified.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import torch
from PIL import Image, ImageDraw, ImageOps
from rich.console import Console
from rich.table import Table
from torchvision import models

from examples.inspect_predictions import font, sha256
from kernelixx.models.loader import load_model
from kernelixx.utils.device import seed_everything


def ranked_classes(logits: torch.Tensor, categories: Sequence[str]) -> list[dict[str, Any]]:
    if logits.shape != (1, len(categories)) or not categories or not torch.isfinite(logits).all():
        raise ValueError("Expected finite scores [1, number of categories]")
    scores, indices = logits.detach().float().softmax(1).topk(min(5, len(categories)), dim=1)
    return [{"index": index, "class": categories[index], "softmax_score": score}
            for index, score in zip(indices[0].tolist(), scores[0].tolist())]


def read_images(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Supply at least one image; do not repeat the same file")
    records, images = [], []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        records.append({"index": index, "image_path": str(path.resolve()), "sha256": sha256(path),
                        "width": image.width, "height": image.height, "predictions": {}})
        images.append(image)
    return records, images


def comparison_sheet(records: list[dict[str, Any]], images: list[Image.Image], path: Path) -> None:
    columns, width, height = 3, 430, 340
    canvas = Image.new("RGB", (columns * width, 65 + math.ceil(len(images) / columns) * height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 13), "Your images: actual model predictions", fill="black", font=font(24))
    for index, (record, image) in enumerate(zip(records, images)):
        x, y = index % columns * width, 65 + index // columns * height
        draw.rectangle((x + 3, y + 3, x + width - 3, y + height - 3), outline="#888888", width=2)
        draw.text((x + 12, y + 9), f"Image #{record['index']}", fill="black", font=font(18))
        thumbnail = ImageOps.contain(image, (400, 210))
        canvas.paste(thumbnail, (x + (width - thumbnail.width) // 2, y + 38 + (210 - thumbnail.height) // 2))
        offset = 260
        for name in ("resnet18", "resnet50"):
            for line in textwrap.wrap(f"{name}: {record['predictions'][name][0]['class']}", 44):
                draw.text((x + 12, y + offset), line, fill="black", font=font(17))
                offset += 21
    canvas.save(path)


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["index", "image_path", "sha256", "resnet18", "resnet50", "resnet18_top5", "resnet50_top5"])
        for record in records:
            predictions = record["predictions"]
            writer.writerow([record["index"], record["image_path"], record["sha256"],
                             predictions["resnet18"][0]["class"], predictions["resnet50"][0]["class"],
                             "; ".join(item["class"] for item in predictions["resnet18"]),
                             "; ".join(item["class"] for item in predictions["resnet50"])])


def predict_images(paths: Sequence[Path], output: Path,
                   cache_dir: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Output exists; choose a new --output directory")
    records, images = read_images(paths)
    seed_everything(42, threads=1)
    model_info = {}
    original_hub = torch.hub.get_dir()
    console = Console()
    try:
        for name, weights in (("resnet18", models.ResNet18_Weights.DEFAULT),
                              ("resnet50", models.ResNet50_Weights.DEFAULT)):
            # Reuse the existing project cache if it contains this checkpoint;
            # otherwise keep torchvision's normal cache/download behavior.
            filename = Path(urlparse(weights.url).path).name
            use_local = cache_dir is not None and (cache_dir / "checkpoints" / filename).is_file()
            torch.hub.set_dir(str(cache_dir.resolve()) if use_local else original_hub)
            model, _ = load_model(name, "default")
            transform = weights.transforms()
            model_info[name] = {"weights": str(weights), "preprocessing": repr(transform),
                                "checkpoint_url": weights.url}
            with torch.inference_mode(), console.status(f"{name}: checking {len(images)} images on CPU..."):
                for record, image in zip(records, images):
                    logits = model(transform(image).unsqueeze(0))
                    record["predictions"][name] = ranked_classes(logits, weights.meta["categories"])
            del model
            gc.collect()
    finally:
        torch.hub.set_dir(original_hub)
    result = {"device": "cpu", "precision": "fp32", "seed": 42, "threads": 1,
              "torch_version": str(torch.__version__), "models": model_info,
              "accuracy_measured": False,
              "note": "No exact ImageNet class labels were supplied. Softmax scores are model scores, "
                      "not measured accuracy or calibrated probabilities. No training or timing experiment.",
              "images": records}
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                                             encoding="utf-8")
    write_csv(records, output / "predictions.csv")
    # Page the contact sheets to keep large input sets from producing a giant bitmap.
    sheets = []
    for start in range(0, len(images), 12):
        sheet = f"comparison-{start // 12 + 1:02d}.png"
        comparison_sheet(records[start:start + 12], images[start:start + 12], output / sheet)
        sheets.append(sheet)
    lines = ["# Your images: ResNet18 versus ResNet50", "", result["note"], "",
             "Both models choose among all 1000 ImageNet classes and use their own official preprocessing.", ""]
    for sheet in sheets:
        lines += [f"![Model answers]({sheet})", ""]
    lines += ["| Image | ResNet18 top answer | ResNet50 top answer |", "|---|---|---|"]
    for record in records:
        predictions = record["predictions"]
        lines.append(f"| {record['index']} | {predictions['resnet18'][0]['class']} | "
                     f"{predictions['resnet50'][0]['class']} |")
    for record in records:
        lines += ["", f"## Image {record['index']}", "", f"Source: `{record['image_path']}`", ""]
        for name, predictions in record["predictions"].items():
            lines += [f"{name} top 5: " + ", ".join(item["class"] for item in predictions), ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = Table(title="Actual top answers (no exact class labels supplied)")
    for title in ("Image", "ResNet18", "ResNet50"):
        table.add_column(title)
    for record in records:
        table.add_row(str(record["index"]), record["predictions"]["resnet18"][0]["class"],
                      record["predictions"]["resnet50"][0]["class"])
    console.print(table)
    console.print(f"Report: {output.resolve() / 'report.md'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="Explicit image files, in display order")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=Path(f"results/user-images/{stamp}-{uuid4().hex[:8]}"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/torch-cache/hub"),
                        help="Optional existing torch hub cache, containing checkpoints/")
    args = parser.parse_args()
    try:
        predict_images(args.images, args.output, args.cache_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

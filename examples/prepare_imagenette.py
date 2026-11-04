"""Prepare a deterministic, class-balanced real-image subset for Kernelixx.

Uses torchvision's official archive downloader and the selected pretrained model's
transforms. Targets are ImageNet-1K indices, not Imagenette's local 0..9 indices.
By default, selects only original ImageNet validation filenames from both Imagenette
folders: Imagenette's own train/val split mixes original ImageNet training images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torchvision import datasets, models

# Names and WNIDs from torchvision.datasets.Imagenette; indices are resolved from
# the actual pretrained weights' metadata instead of assuming local class order.
WNID_TO_CATEGORY = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
ARCHIVE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
ARCHIVE_MD5 = "e793b78cc4c9e9a4ccc0c1155377a412"


def imagenet_targets(wnids: Sequence[str], categories: Sequence[str]) -> dict[str, int]:
    if set(wnids) != set(WNID_TO_CATEGORY) or len(wnids) != 10:
        raise ValueError("Expected exactly the ten Imagenette WNID folders")
    targets = {}
    names = list(categories)
    for wnid in wnids:
        name = WNID_TO_CATEGORY[wnid]
        if names.count(name) != 1:
            raise ValueError(f"Expected one ImageNet class named {name!r}")
        targets[wnid] = names.index(name)
    return targets


def balanced_indices(targets: Sequence[int], samples: int, seed: int) -> list[int]:
    if samples < 10 or samples % 10:
        raise ValueError("samples must be a positive multiple of ten")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    values = np.asarray(targets)
    if set(values.tolist()) != set(range(10)):
        raise ValueError("Expected local dataset labels 0..9")
    generator = np.random.default_rng(seed)
    selected = []
    per_class = samples // 10
    for label in range(10):
        indices = np.flatnonzero(values == label)
        if len(indices) < per_class:
            raise ValueError(f"Not enough examples in class {label}: need {per_class}, found {len(indices)}")
        selected.extend(generator.choice(indices, size=per_class, replace=False).tolist())
    generator.shuffle(selected)
    return selected


def prepare_imagenette(root: Path, output: Path, samples: int = 200, seed: int = 42,
                       model: str = "resnet18", download: bool = False,
                       split: str = "imagenet-val") -> dict[str, Any]:
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must have the .npz extension")
    metadata_path = Path(str(output) + ".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Output already exists; choose a new --output path to preserve it")
    if model not in {"resnet18", "resnet50"}:
        raise ValueError("model must be resnet18 or resnet50")
    if split not in {"imagenet-val", "imagenette-val"}:
        raise ValueError("split must be imagenet-val or imagenette-val")
    if samples < 10 or samples % 10 or not 0 <= seed < 2**32:
        raise ValueError("samples must be a positive multiple of ten; seed must be in [0, 2**32)")
    weights = (models.ResNet18_Weights.DEFAULT if model == "resnet18"
               else models.ResNet50_Weights.DEFAULT)
    if download:
        # torchvision checks the archive MD5 before extraction on a fresh download.
        datasets.Imagenette(root=str(root), split="val", size="160px", download=True)
    image_root = root / "imagenette2-160"
    source_splits = ("train", "val") if split == "imagenet-val" else ("val",)
    folders = []
    for source_split in source_splits:
        directory = image_root / source_split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {directory}; use --download to fetch Imagenette")
        folders.append(datasets.ImageFolder(str(directory)))
    classes = folders[0].classes
    if any(folder.classes != classes for folder in folders):
        raise ValueError("Source folders do not have matching class mappings")
    mapping = imagenet_targets(classes, weights.meta["categories"])
    catalog = [item for folder in folders for item in folder.samples
               if split == "imagenette-val" or Path(item[0]).match("ILSVRC2012_val_*.JPEG")]
    selected = balanced_indices([label for _, label in catalog], samples, seed)
    transform = weights.transforms()
    torch.set_num_threads(1)
    inputs = np.empty((samples, 3, 224, 224), dtype=np.float32)
    labels = np.empty(samples, dtype=np.int64)
    manifest = []
    for position, index in enumerate(selected):
        image_path, local_label = catalog[index]
        image = transform(folders[0].loader(image_path))
        wnid = classes[local_label]
        inputs[position] = image.numpy()
        labels[position] = mapping[wnid]
        path = Path(image_path)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest.append({"path": path.relative_to(image_root).as_posix(),
                         "sha256": digest, "wnid": wnid, "imagenet_target": mapping[wnid]})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing datasets from accidental replacement.
    with output.open("xb") as target:
        np.savez_compressed(target, inputs=inputs, labels=labels)
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metadata: dict[str, Any] = {
        "dataset": "Imagenette", "split": split, "source_splits": list(source_splits),
        "eligible_images": len(catalog), "image_size_variant": "160px",
        "source": "https://github.com/fastai/imagenette", "archive_url": ARCHIVE_URL,
        "archive_expected_md5": ARCHIVE_MD5,
        "archive_integrity_note": "torchvision verifies MD5 on download; an existing extracted tree is not re-verified.",
        "model": model, "weights": str(weights), "preprocessing": repr(weights.transforms()),
        "samples": samples, "samples_per_class": samples // 10, "seed": seed,
        "target_space": "Original ImageNet-1K class indices; argmax remains over all 1000 outputs.",
        "selection": "Seeded, without replacement, class-balanced; no prediction-based filtering.",
        "filename_filter": "ILSVRC2012_val_*.JPEG" if split == "imagenet-val" else None,
        "pretraining_overlap_note": "Only original ImageNet validation filenames selected."
        if split == "imagenet-val" else
        "Imagenette val contains original ImageNet training filenames; overlap with pretraining is not excluded.",
        "class_mapping": {wnid: {"name": WNID_TO_CATEGORY[wnid], "imagenet_target": index}
                          for wnid, index in mapping.items()},
        "input_shape": list(inputs.shape), "inputs_dtype": str(inputs.dtype),
        "labels_dtype": str(labels.dtype), "npz_sha256": digest, "manifest": manifest,
        "limitations": "A small, ten-class validation subset; accuracy does not represent the full ImageNet benchmark or production data.",
    }
    with metadata_path.open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/datasets/imagenette"))
    parser.add_argument("--output", type=Path, default=Path("results/datasets/imagenet-val-resnet18-200.npz"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--split", choices=("imagenet-val", "imagenette-val"), default="imagenet-val",
                        help="Default excludes original ImageNet training filenames")
    parser.add_argument("--download", action="store_true", help="Download the official 160px archive (about 100 MB)")
    args = parser.parse_args()
    try:
        metadata = prepare_imagenette(args.root, args.output, args.samples, args.seed, args.model, args.download, args.split)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Prepared {metadata['samples']} real validation images: {args.output}")
    print(f"ImageNet labels and preprocessing recorded in {args.output}.json")


if __name__ == "__main__":
    main()

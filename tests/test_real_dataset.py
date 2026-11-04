import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights

from examples.prepare_imagenette import WNID_TO_CATEGORY, balanced_indices, imagenet_targets, prepare_imagenette
from examples.validate_batching import check_batch_equivalence
from kernelixx.core.optimizer import ExperimentConfig


def test_imagenette_targets_are_original_imagenet_indices():
    mapping = imagenet_targets(sorted(WNID_TO_CATEGORY), ResNet18_Weights.DEFAULT.meta["categories"])
    assert mapping == {"n01440764": 0, "n02102040": 217, "n02979186": 482,
                       "n03000684": 491, "n03028079": 497, "n03394916": 566,
                       "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}


def test_balanced_sampling_is_repeatable_and_without_replacement():
    targets = np.repeat(np.arange(10), 5).tolist()
    first = balanced_indices(targets, 20, 42)
    assert first == balanced_indices(targets, 20, 42)
    assert first != balanced_indices(targets, 20, 43)
    assert len(set(first)) == 20
    assert np.bincount(np.asarray(targets)[first]).tolist() == [2] * 10


@pytest.mark.parametrize("samples", [0, 9, 11, 100])
def test_invalid_or_insufficient_sample_counts(samples):
    with pytest.raises(ValueError):
        balanced_indices(list(range(10)), samples, 42)


def test_prepare_images_preprocess_and_npz_contract_without_network(tmp_path):
    for index, wnid in enumerate(WNID_TO_CATEGORY):
        for split in ("train", "val"):
            directory = tmp_path / "imagenette2-160" / split / wnid
            directory.mkdir(parents=True)
            # Both Imagenette folders may contain original ImageNet val images.
            suffix = index * 2 + (split == "val")
            Image.new("L", (40, 32), color=128).save(directory / f"ILSVRC2012_val_{suffix:08d}.JPEG")
            # This original training filename must never enter the default subset.
            Image.new("L", (40, 32), color=0).save(directory / f"{wnid}_123.JPEG")
    path = tmp_path / "prepared.npz"
    metadata = prepare_imagenette(tmp_path, path, samples=10)
    with np.load(path, allow_pickle=False) as data:
        assert data["inputs"].shape == (10, 3, 224, 224)
        assert data["inputs"].dtype == np.float32
        assert data["labels"].dtype == np.int64
        assert sorted(data["labels"].tolist()) == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        expected = (128 / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        np.testing.assert_allclose(data["inputs"][0, :, 0, 0], expected, atol=1e-6)
    assert len(metadata["manifest"]) == 10
    assert metadata["eligible_images"] == 20
    assert metadata["split"] == "imagenet-val"
    assert all(item["path"].split("/")[-1].startswith("ILSVRC2012_val_") for item in metadata["manifest"])
    assert {item["path"].split("/")[0] for item in metadata["manifest"]} == {"train", "val"}
    assert all(len(item["sha256"]) == 64 for item in metadata["manifest"])
    assert json.loads(path.with_suffix(".npz.json").read_text())["npz_sha256"] == metadata["npz_sha256"]
    with pytest.raises(FileExistsError):
        prepare_imagenette(tmp_path, path, samples=10)


@pytest.mark.parametrize("batch_dependent", [False, True])
def test_batch_equivalence_keeps_partial_batch_and_detects_changed_behavior(tmp_path, monkeypatch, batch_dependent):
    class Probe(nn.Module):
        def forward(self, inputs):
            score = inputs.mean(1)
            offset = inputs.shape[0] - 1 if batch_dependent else 0
            return torch.stack((score + offset, score + 1), dim=1)

    monkeypatch.setattr("examples.validate_batching.load_model",
                        lambda *args, **kwargs: (Probe(), {"name": "batch-probe"}))
    path = tmp_path / "data.npz"
    np.savez(path, inputs=np.zeros((3, 16), np.float32), labels=np.ones(3, np.int64))
    result = check_batch_equivalence(ExperimentConfig(device="cpu", input_shape=(1, 16), dataset=str(path)), (1, 2))
    validation = result["validation"]
    assert validation["validation_examples"] == 3
    assert validation["baseline_correct"] == 3
    assert validation["optimized_correct"] == (1 if batch_dependent else 3)
    assert validation["passed"] is (not batch_dependent)

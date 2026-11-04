import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from examples.compare_predictions import read_review
from examples.inspect_predictions import checked_manifest, inspect_predictions, prediction_rows
from examples.prepare_imagenette import WNID_TO_CATEGORY, prepare_imagenette


def test_photo_predictions_preserve_order_and_use_all_output_classes():
    paths = [Path("second.JPEG"), Path("first.JPEG"), Path("third.JPEG")]
    logits = torch.tensor([[0., 0., 9.], [9., 0., 0.], [0., 9., 0.]])
    alternate = torch.tensor([[0., 0., 9.], [0., 9., 0.], [0., 9., 0.]])
    rows = prediction_rows(logits, [0, 0, 1], ["fish", "dog", "unseen class"], paths, alternate)
    assert [row["image_path"] for row in rows] == [str(path) for path in paths]
    assert [row["correct"] for row in rows] == [False, True, True]
    assert rows[0]["predicted"] == "unseen class"  # Never restrict argmax to supplied labels.
    assert rows[1]["onnx_matches_pytorch"] is False
    with pytest.raises(ValueError, match="finite logits"):
        prediction_rows(logits * float("nan"), [0, 0, 1], ["a", "b", "c"], paths)


def test_manifest_detects_label_order_changed_image_and_path_escape(tmp_path):
    image = tmp_path / "photo.JPEG"
    image.write_bytes(b"original image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    metadata = {"samples": 1, "manifest": [{"path": image.name, "imagenet_target": 7, "sha256": digest}]}
    assert checked_manifest(metadata, [7], tmp_path) == [image]
    with pytest.raises(ValueError, match="label order"):
        checked_manifest(metadata, [8], tmp_path)
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_manifest(metadata, [7], tmp_path)
    metadata["manifest"][0]["path"] = "../outside.JPEG"
    with pytest.raises(ValueError, match="escapes"):
        checked_manifest(metadata, [7], tmp_path)


def test_review_end_to_end_saved_results_images_and_no_overwrite(tmp_path, monkeypatch):
    for index, wnid in enumerate(WNID_TO_CATEGORY):
        for split in ("train", "val"):
            directory = tmp_path / "imagenette2-160" / split / wnid
            directory.mkdir(parents=True)
            Image.new("RGB", (40, 32), color=(index * 20, 100, 80)).save(
                directory / f"ILSVRC2012_val_{index:08d}.JPEG")
    dataset = tmp_path / "prepared.npz"
    prepare_imagenette(tmp_path, dataset, samples=10)

    class PredictFish(nn.Module):
        def forward(self, inputs):
            result = torch.zeros((len(inputs), 1000))
            result[:, 0] = 1
            return result

    monkeypatch.setattr("examples.inspect_predictions.load_model", lambda *args: (PredictFish(), {}))
    monkeypatch.setattr("examples.inspect_predictions.OnnxRunner", lambda *args, **kwargs: PredictFish())
    onnx = tmp_path / "probe.onnx"
    onnx.write_bytes(b"test model identity")
    output = tmp_path / "review"
    summary = inspect_predictions(dataset, tmp_path / "imagenette2-160", output, onnx)
    assert (summary["examples"], summary["correct"], summary["errors"]) == (10, 1, 9)
    assert summary["onnx_changed_predictions"] == 0
    assert summary["onnx_validation"]["passed"] is True
    result = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert len(result["predictions"]) == 10
    csv_bytes = (output / "predictions.csv").read_bytes()
    assert b"\r\r\n" not in csv_bytes
    assert len((output / "errors.csv").read_text(encoding="utf-8-sig").splitlines()) == 10
    with Image.open(output / "errors-01.png") as sheet:
        assert sheet.size[0] == 1320
        assert sheet.size[1] > 320
    assert (output / "correct-examples.png").is_file()
    assert "## Every photo" in (output / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        inspect_predictions(dataset, tmp_path / "imagenette2-160", output)
    # The comparison reader independently checks saved answers against their source manifest.
    loaded_summary, loaded_rows = read_review(output / "predictions.json", "resnet18")
    assert loaded_summary["correct"] == 1
    assert len(loaded_rows[0]["image_sha256"]) == 64
    with pytest.raises(ValueError, match="Expected a resnet50"):
        read_review(output / "predictions.json", "resnet50")
    result["predictions"][0]["correct"] = not result["predictions"][0]["correct"]
    inconsistent = tmp_path / "inconsistent.json"
    inconsistent.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        read_review(inconsistent, "resnet18")
    # A modified NPZ must never silently become associated with the original manifest.
    with dataset.open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="dataset hash"):
        inspect_predictions(dataset, tmp_path / "imagenette2-160", tmp_path / "another-review")
    assert not (tmp_path / "another-review").exists()

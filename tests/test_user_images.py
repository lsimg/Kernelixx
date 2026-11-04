import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from examples.predict_images import predict_images, ranked_classes, read_images


def test_user_image_rankings_use_all_categories_without_forcing_an_expected_answer():
    logits = torch.tensor([[0., 4., 1.]])
    ranked = ranked_classes(logits, ["apple", "pepper", "banana"])
    assert [item["class"] for item in ranked] == ["pepper", "banana", "apple"]
    assert sum(item["softmax_score"] for item in ranked) == pytest.approx(1)
    with pytest.raises(ValueError):
        ranked_classes(logits * float("nan"), ["a", "b", "c"])


def test_user_images_keep_explicit_order_and_support_unicode_grayscale_and_alpha(tmp_path):
    paths = [tmp_path / "яблоко.png", tmp_path / "другое.png"]
    Image.new("L", (20, 30), color=128).save(paths[0])
    Image.new("RGBA", (40, 25), color=(255, 0, 0, 255)).save(paths[1])
    records, images = read_images(paths)
    assert [item["image_path"] for item in records] == [str(path) for path in paths]
    assert [image.mode for image in images] == ["RGB", "RGB"]
    assert [(item["width"], item["height"]) for item in records] == [(20, 30), (40, 25)]
    assert all(len(item["sha256"]) == 64 for item in records)
    with pytest.raises(ValueError, match="repeat"):
        read_images([paths[0], paths[0]])


def test_user_image_inference_reports_and_restores_model_cache(tmp_path, monkeypatch):
    path = tmp_path / "input.png"
    Image.new("RGB", (60, 40), color="red").save(path)
    cache = tmp_path / "cache"
    (cache / "checkpoints").mkdir(parents=True)
    (cache / "checkpoints" / "resnet50-11ad3fa6.pth").write_bytes(b"test cache marker")
    original_hub = torch.hub.get_dir()

    class Probe(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def forward(self, inputs):
            assert inputs.shape == (1, 3, 224, 224)
            assert not torch.is_grad_enabled()
            result = torch.zeros((1, 1000))
            result[:, self.index] = 10
            return result

    def loader(name, weights):
        assert weights == "default"
        if name == "resnet50":
            assert Path(torch.hub.get_dir()) == cache
        else:
            assert torch.hub.get_dir() == original_hub
        return Probe(948 if name == "resnet50" else 954), {}

    monkeypatch.setattr("examples.predict_images.load_model", loader)
    output = tmp_path / "output"
    result = predict_images([path], output, cache)
    assert torch.hub.get_dir() == original_hub
    assert result["accuracy_measured"] is False
    assert result["images"][0]["predictions"]["resnet50"][0]["class"] == "Granny Smith"
    assert len(json.loads((output / "predictions.json").read_text())["images"]) == 1
    assert "top 5" in (output / "report.md").read_text(encoding="utf-8")
    assert b"\r\r\n" not in (output / "predictions.csv").read_bytes()
    with Image.open(output / "comparison-01.png") as sheet:
        assert sheet.width == 1290
    with pytest.raises(FileExistsError):
        predict_images([path], output, cache)

    monkeypatch.setattr("examples.predict_images.load_model",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("load failed")))
    with pytest.raises(RuntimeError, match="load failed"):
        predict_images([path], tmp_path / "failed-output", cache)
    assert torch.hub.get_dir() == original_hub
    assert not (tmp_path / "failed-output").exists()

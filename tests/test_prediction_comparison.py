import csv
import json

import pytest
from PIL import Image

from examples.compare_predictions import compare_reviews, compare_rows


def probe_rows(tmp_path):
    left, right = [], []
    # Both correct, corrected, newly wrong, both wrong with different answers.
    for index, (a, b) in enumerate(((0, 0), (1, 0), (0, 1), (1, 2)), start=1):
        path = tmp_path / f"{index}.JPEG"
        Image.new("RGB", (30, 20), color="blue").save(path)
        common = {"index": index, "image_path": str(path), "source_path": path.name,
                  "image_sha256": str(index), "expected_index": 0, "expected": "fish"}
        names = ["fish", "dog", "cat"]
        left.append({**common, "predicted_index": a, "predicted": names[a], "correct": a == 0})
        right.append({**common, "predicted_index": b, "predicted": names[b], "correct": b == 0})
    return left, list(reversed(right))


def test_comparison_matches_photos_not_row_positions_and_keeps_all_four_outcomes(tmp_path):
    left, right = probe_rows(tmp_path)
    result = compare_rows(left, right)
    assert [row["index"] for row in result] == [1, 2, 3, 4]
    assert [row["outcome"] for row in result] == ["both_correct", "fixed", "regressed", "both_wrong"]
    assert result[3]["resnet18"] == "dog"
    assert result[3]["resnet50"] == "cat"


@pytest.mark.parametrize("mismatch", ["hash", "label", "duplicate", "missing"])
def test_comparison_rejects_incompatible_photo_sets(tmp_path, mismatch):
    left, right = probe_rows(tmp_path)
    if mismatch == "hash":
        right[0]["image_sha256"] = "changed"
    elif mismatch == "label":
        right[0]["expected_index"] = 1
    elif mismatch == "duplicate":
        right.append(right[0])
    else:
        right.pop()
    with pytest.raises(ValueError):
        compare_rows(left, right)


def test_comparison_reports_fixes_and_regressions_instead_of_only_net_gain(tmp_path, monkeypatch):
    left, right = probe_rows(tmp_path)
    paths = [tmp_path / "resnet18.json", tmp_path / "resnet50.json"]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("examples.compare_predictions.read_review",
                        lambda path, model: ({"model": model}, left if model == "resnet18" else right))
    output = tmp_path / "comparison"
    summary = compare_reviews(*paths, output)
    assert summary["counts"] == {"fixed": 1, "regressed": 1, "both_wrong": 1, "both_correct": 1}
    assert summary["accuracy_difference_percentage_points"] == 0
    assert summary["resnet18_correct"] == summary["resnet50_correct"] == 2
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert len(result["predictions"]) == 4
    with (output / "fixed.csv").open(encoding="utf-8-sig", newline="") as source:
        fixed = list(csv.DictReader(source))
    assert [row["index"] for row in fixed] == ["2"]
    assert b"\r\r\n" not in (output / "comparison.csv").read_bytes()
    for name in ("fixed-01.png", "regressed-01.png", "both_wrong-01.png"):
        with Image.open(output / name) as sheet:
            assert sheet.width == 1320
    assert "Net gain: **+0**" in (output / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        compare_reviews(*paths, output)

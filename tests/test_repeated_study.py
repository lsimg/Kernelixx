import json

import pytest

from examples.repeat_batching import main, summarize_pairs


def test_paired_summary_uses_whole_trial_ratios():
    records = [{"trial": trial, "batch_size": batch, "images_per_second": rate}
               for trial, a, b in [(1, 50, 200), (2, 100, 300), (3, 100, 500)]
               for batch, rate in [(1, a), (8, b)]]
    result = summarize_pairs(records)
    assert result["paired_throughput_ratios"] == [4, 3, 5]
    assert result["paired_ratio_statistics"]["median"] == 4
    assert result["paired_ratio_statistics"]["sample_standard_deviation"] == 1
    assert result["trials_with_batch8_higher_throughput"] == 3


@pytest.mark.parametrize("bad_rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rates_cannot_enter_summary(bad_rate):
    records = [{"trial": trial, "batch_size": batch, "images_per_second": bad_rate if trial == 1 else 10}
               for trial in (1, 2) for batch in (1, 8)]
    with pytest.raises(ValueError):
        summarize_pairs(records)


def test_missing_or_duplicate_pairs_are_rejected():
    records = [{"trial": 1, "batch_size": 1, "images_per_second": 10},
               {"trial": 2, "batch_size": 1, "images_per_second": 10}]
    with pytest.raises(ValueError, match="complete"):
        summarize_pairs(records)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_pairs(records + [records[0]])


def test_read_summary_does_not_launch_benchmarks(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reading results must not launch a new experiment")
    monkeypatch.setattr("examples.repeat_batching.run_child", forbidden)
    records = [{"trial": trial, "batch_size": batch, "images_per_second": rate}
               for trial in (1, 2) for batch, rate in ((1, 50), (8, 200))]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": 1, "experiment": "batching-repeat-study",
                                "model": "resnet50", "device": "cuda:0", "records": records}))
    main(["--read-summary", str(tmp_path)])
    assert "4.000x" in capsys.readouterr().out

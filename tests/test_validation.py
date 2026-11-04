import pytest
import torch

from kernelixx.core.validator import Tolerances, checked_output, validate


def test_known_errors_and_accuracy():
    reference = [torch.tensor([[1.0, 0.0], [0.0, 2.0]])]
    candidate = [torch.tensor([[1.1, 0.0], [0.0, 2.1]])]
    result = validate(reference, candidate, Tolerances(atol=0.11, rtol=0), [torch.tensor([0, 1])])
    assert result["passed"]
    assert result["maximum_absolute_error"] == pytest.approx(0.1, abs=1e-6)
    assert result["mean_absolute_error"] == pytest.approx(0.05, abs=1e-6)
    assert result["maximum_relative_error"] == pytest.approx(0.1, abs=1e-6)
    assert result["class_agreement"] == 1
    assert result["baseline_accuracy"] == result["optimized_accuracy"] == 1
    assert result["baseline_correct"] == result["optimized_correct"] == 2


def test_elementwise_rule_and_no_accuracy_claim():
    result = validate([torch.tensor([[0.0, 1.0]])], [torch.tensor([[0.002, 1.0]])], Tolerances())
    assert not result["passed"]
    assert result["maximum_relative_error"] > 1000
    assert not result["accuracy_measured"]
    assert result["baseline_accuracy"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_is_rejected(bad):
    result = validate([torch.ones(1, 2)], [torch.tensor([[bad, 1.0]])], Tolerances())
    assert not result["passed"]
    assert result["maximum_absolute_error"] is None


def test_shape_mismatch():
    assert not validate([torch.ones(1, 2)], [torch.ones(1, 3)], Tolerances())["passed"]


def test_matrix_regression_does_not_claim_class_predictions():
    result = validate([torch.ones(2, 3)], [torch.ones(2, 3)], Tolerances(), classification=False)
    assert result["passed"] and result["class_agreement"] is None


def test_double_output_differences_are_not_rounded_away():
    result = validate([torch.tensor([[1.0]], dtype=torch.float64)],
                      [torch.tensor([[1.0 + 1e-10]], dtype=torch.float64)], Tolerances(0, 0))
    assert not result["passed"]


@pytest.mark.parametrize("atol,rtol", [(-1, 0), (0, -1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_tolerances(atol, rtol):
    with pytest.raises(ValueError):
        Tolerances(atol, rtol)


def test_invalid_labels_and_output_contract():
    with pytest.raises(ValueError, match="class range"):
        validate([torch.ones(1, 2)], [torch.ones(1, 2)], Tolerances(), [torch.tensor([5])])
    with pytest.raises(TypeError, match="single Tensor"):
        checked_output({"logits": torch.ones(1, 2)})

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from kernelixx.core.benchmark import Runner


@dataclass(frozen=True)
class Tolerances:
    atol: float = 1e-3
    rtol: float = 1e-2

    def __post_init__(self) -> None:
        if not np.isfinite([self.atol, self.rtol]).all() or min(self.atol, self.rtol) < 0:
            raise ValueError("atol and rtol must be finite and nonnegative")


def checked_output(output: Any) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise TypeError("v1 requires a single Tensor output, not tuple/dict outputs")
    if output.ndim < 1 or output.numel() == 0 or not output.is_floating_point():
        raise ValueError("Model output must be a nonempty floating-point Tensor with a batch dimension")
    return output.detach().cpu()


def validate(reference: list[torch.Tensor], candidate: list[torch.Tensor],
             tolerances: Tolerances, labels: list[torch.Tensor] | None = None,
             classification: bool = True) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("Validation needs matching nonempty output batches")
    if labels is not None and len(labels) != len(reference):
        raise ValueError("Labels must match all validation batches")
    max_abs = max_rel = sum_abs = 0.0
    elements = agreement = examples = base_correct = opt_correct = 0
    passed, finite, all_classification = True, True, classification
    for index, (ref, out) in enumerate(zip(reference, candidate)):
        ref, out = checked_output(ref), checked_output(out)
        if ref.shape != out.shape:
            return {"passed": False, "reason": "Output shapes differ",
                    "reference_shape": list(ref.shape), "candidate_shape": list(out.shape),
                    "accuracy_measured": False}
        # Compare in float64 so validation does not hide changes in double outputs.
        ref, out = ref.double(), out.double()
        batch_finite = bool(torch.isfinite(ref).all() and torch.isfinite(out).all())
        finite &= batch_finite
        if batch_finite:
            error = (ref - out).abs()
            max_abs = max(max_abs, error.max().item())
            max_rel = max(max_rel, (error / ref.abs().clamp_min(1e-8)).max().item())
            sum_abs += error.sum(dtype=torch.float64).item()
            elements += ref.numel()
            passed &= bool(torch.all(error <= tolerances.atol + tolerances.rtol * ref.abs()))
        else:
            passed = False
        is_classification = classification and ref.ndim == 2 and ref.shape[1] > 1
        all_classification &= is_classification
        if is_classification and batch_finite:
            rp, op = ref.argmax(1), out.argmax(1)
            examples += ref.shape[0]
            agreement += (rp == op).sum().item()
            if labels is not None:
                target = labels[index].cpu()
                if target.shape != rp.shape or target.dtype != torch.int64:
                    raise ValueError("Classification labels must be int64 of shape [N]")
                if bool(((target < 0) | (target >= ref.shape[1])).any()):
                    raise ValueError("Labels outside the model's class range")
                base_correct += (rp == target).sum().item()
                opt_correct += (op == target).sum().item()
        elif labels is not None:
            raise ValueError("Accuracy requires finite [N, classes] logits")
    measured = labels is not None
    return {"passed": passed and finite, "finite": finite,
            "maximum_absolute_error": max_abs if finite else None,
            "mean_absolute_error": sum_abs / elements if finite else None,
            "maximum_relative_error": max_rel if finite else None,
            "relative_error_denominator_floor": 1e-8,
            "class_agreement": agreement / examples if all_classification and examples else None,
            "accuracy_measured": measured,
            "baseline_accuracy": base_correct / examples if measured else None,
            "optimized_accuracy": opt_correct / examples if measured else None,
            "baseline_correct": base_correct if measured else None,
            "optimized_correct": opt_correct if measured else None,
            "validation_examples": sum(int(x.shape[0]) for x in reference),
            "atol": tolerances.atol, "rtol": tolerances.rtol,
            "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
            "accuracy_note": "Accuracy on supplied labels only." if measured else
            "Real task accuracy was not measured. Output similarity is not accuracy."}


def outputs(runner: Runner, batches: list[torch.Tensor]) -> list[torch.Tensor]:
    result = []
    dtype = torch.float16 if runner.precision == "fp16" else torch.float32
    with torch.inference_mode():
        for batch in batches:
            output = checked_output(runner(batch.to(runner.device, dtype=dtype)))
            if output.shape[0] != batch.shape[0]:
                raise ValueError("Output batch dimension must match the input batch")
            result.append(output)
    return result

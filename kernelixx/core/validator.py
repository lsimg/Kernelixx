from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from itertools import zip_longest

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


class ValidationAccumulator:
    """Exact scalar aggregation; scratch tensors are bounded by chunk_elements."""
    def __init__(self, tolerances: Tolerances, classification: bool = True, chunk_elements: int = 65536):
        if chunk_elements < 1:
            raise ValueError("chunk-elements must be positive")
        self.tolerances, self.classification, self.chunk_elements = tolerances, classification, chunk_elements
        self.max_abs = self.max_rel = self.sum_abs = 0.0
        self.elements = self.agreement = self.examples = self.base_correct = self.opt_correct = 0
        self.validation_examples = self.batches = 0
        self.passed = self.finite = True
        self.all_classification = classification
        self.measured: bool | None = None
        self.failure = None

    def update(self, reference: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        ref, out = checked_output(reference), checked_output(candidate)
        self.batches += 1
        self.validation_examples += ref.shape[0]
        measured = labels is not None
        if self.measured is not None and measured != self.measured:
            raise ValueError("Labels must match all validation batches")
        self.measured = measured
        if ref.shape != out.shape:
            self.passed = False
            self.failure = {"reason": "Output shapes differ", "reference_shape": list(ref.shape),
                            "candidate_shape": list(out.shape)}
            return
        batch_finite = True
        # Flatten may copy a non-contiguous single batch, never an entire dataset.
        rf, of = ref.reshape(-1), out.reshape(-1)
        for start in range(0, ref.numel(), self.chunk_elements):
            r, o = rf[start:start + self.chunk_elements].double(), of[start:start + self.chunk_elements].double()
            finite = bool(torch.isfinite(r).all() and torch.isfinite(o).all())
            batch_finite &= finite
            if not finite:
                self.passed = False
                continue
            error = (r - o).abs()
            self.max_abs = max(self.max_abs, error.max().item())
            self.max_rel = max(self.max_rel, (error / r.abs().clamp_min(1e-8)).max().item())
            self.sum_abs += error.sum(dtype=torch.float64).item()
            self.elements += r.numel()
            self.passed &= bool(torch.all(error <= self.tolerances.atol + self.tolerances.rtol * r.abs()))
        self.finite &= batch_finite
        classification = self.classification and ref.ndim == 2 and ref.shape[1] > 1
        self.all_classification &= classification
        if classification and batch_finite:
            rp, op = ref.argmax(1), out.argmax(1)
            self.examples += ref.shape[0]
            self.agreement += (rp == op).sum().item()
            if labels is not None:
                target = labels.cpu()
                if target.shape != rp.shape or target.dtype != torch.int64:
                    raise ValueError("Classification labels must be int64 of shape [N]")
                if bool(((target < 0) | (target >= ref.shape[1])).any()):
                    raise ValueError("Labels outside the model's class range")
                self.base_correct += (rp == target).sum().item()
                self.opt_correct += (op == target).sum().item()
        elif labels is not None:
            raise ValueError("Accuracy requires finite [N, classes] logits")

    def result(self) -> dict[str, Any]:
        if not self.batches:
            raise ValueError("Validation needs matching nonempty output batches")
        if self.failure:
            return {"passed": False, **self.failure, "accuracy_measured": False}
        measured = bool(self.measured)
        return {"passed": self.passed and self.finite, "finite": self.finite,
                "maximum_absolute_error": self.max_abs if self.finite else None,
                "mean_absolute_error": self.sum_abs / self.elements if self.finite else None,
                "maximum_relative_error": self.max_rel if self.finite else None,
                "relative_error_denominator_floor": 1e-8,
                "class_agreement": self.agreement / self.examples if self.all_classification and self.examples else None,
                "accuracy_measured": measured,
                "baseline_accuracy": self.base_correct / self.examples if measured else None,
                "optimized_accuracy": self.opt_correct / self.examples if measured else None,
                "baseline_correct": self.base_correct if measured else None,
                "optimized_correct": self.opt_correct if measured else None,
                "accuracy_drop": (self.base_correct - self.opt_correct) / self.examples if measured else None,
                "validation_examples": self.validation_examples,
                "atol": self.tolerances.atol, "rtol": self.tolerances.rtol,
                "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
                "accuracy_note": "Accuracy on supplied labels only." if measured else
                "Real task accuracy was not measured. Output similarity is not accuracy."}


def validate(reference: Iterable[torch.Tensor], candidate: Iterable[torch.Tensor],
             tolerances: Tolerances, labels: Iterable[torch.Tensor] | None = None,
             classification: bool = True, chunk_elements: int = 65536) -> dict[str, Any]:
    accumulator = ValidationAccumulator(tolerances, classification, chunk_elements)
    sentinel = object()
    streams = (reference, candidate) if labels is None else (reference, candidate, labels)
    for values in zip_longest(*streams, fillvalue=sentinel):
        if any(v is sentinel for v in values):
            raise ValueError("Validation needs matching nonempty output batches and labels")
        accumulator.update(*values)
    return accumulator.result()


def run_output(runner: Runner, batch: torch.Tensor) -> torch.Tensor:
    dtype = torch.float16 if runner.precision == "fp16" else torch.float32
    with torch.inference_mode():
        output = checked_output(runner(batch.to(runner.device, dtype=dtype)))
    if output.shape[0] != batch.shape[0]:
        raise ValueError("Output batch dimension must match the input batch")
    return output


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

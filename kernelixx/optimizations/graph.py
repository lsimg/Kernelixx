from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from kernelixx.optimizations import UnsupportedOptimization


class OnnxRunner:
    precision = "fp32"
    backend = "onnxruntime/cpu"
    device = torch.device("cpu")

    def __init__(self, path: Path, threads: int):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        graph = onnx.load(str(path))
        self.weight_bytes = sum(numpy_helper.to_array(t).nbytes for t in graph.graph.initializer)
        self.serialized_bytes = path.stat().st_size
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # CPU Tensor.numpy shares input storage. ORT dispatch/output wrapping is timed.
        output = self.session.run(None, {self.input_name: inputs.detach().numpy()})
        if len(output) != 1:
            raise ValueError("v1 ONNX contract requires one Tensor output")
        return torch.from_numpy(output[0])


def export_onnx(model: nn.Module, sample: torch.Tensor, device: torch.device,
                directory: Path, threads: int) -> tuple[OnnxRunner, dict[str, object]]:
    if device.type != "cpu":
        raise UnsupportedOptimization("v1 ONNX backend is CPU-only; use --device cpu for a fair comparison")
    for module in ("onnx", "onnxruntime"):
        if importlib.util.find_spec(module) is None:
            raise UnsupportedOptimization("ONNX dependencies missing; install kernelixx[onnx]")
    import onnx

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.onnx"
    # Legacy exporter is intentional: avoids an unused onnxscript dependency.
    torch.onnx.export(model.cpu().eval(), (sample.cpu(),), str(path), dynamo=False,
                      opset_version=17, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    onnx.checker.check_model(str(path))
    return OnnxRunner(path, threads), {"export_path": str(path), "opset": 17,
                                      "graph_optimization": "ORT_ENABLE_ALL",
                                      "note": "Export and session creation excluded from steady-state latency."}

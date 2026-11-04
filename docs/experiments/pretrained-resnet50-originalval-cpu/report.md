# Kernelixx measured experiment

Experiment: inference

## Environment

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "processor": "AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD",
  "versions": {
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "numpy": "2.2.6",
    "onnx": "1.23.2",
    "onnxruntime": "1.31.1",
    "rich": "14.2.0"
  },
  "torch_runtime_version": "2.9.1+cu128",
  "driver_visible_gpus": [
    {
      "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "driver": "591.91",
      "total_memory_mib": 8188
    }
  ],
  "cuda_available": true,
  "torch_cuda": "12.8",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "total_memory_bytes": 8585216000,
    "compute_capability": "8.9"
  },
  "cpu_threads": 1,
  "quantized_engines": [
    "onednn"
  ]
}
```

## Configuration

```json
{
  "model": "resnet50",
  "weights": "default",
  "device": "cpu",
  "input_shape": [
    1,
    3,
    224,
    224
  ],
  "seed": 42,
  "threads": 1,
  "validation_batches": 3,
  "dataset": "results/datasets/imagenet-val-resnet50-200.npz",
  "objective": "latency",
  "min_improvement": 0.01,
  "max_accuracy_drop": 0.01,
  "task": "auto"
}
```

## Model

```json
{
  "name": "resnet50",
  "architecture": "ResNet",
  "weights": "default",
  "parameters": 25557032,
  "trainable_parameters": 25557032
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ok | 216.4447 | 237.6677 | 4.5844 | N/A | 102441032 | N/A |
| onnx | ok | 131.8114 | 138.9689 | 7.6232 | N/A | 102121888 | 1.6421 |
| quantization | rejected | 221.0995 | 241.5438 | 4.4953 | N/A | 96297044 | 0.9789 |

## Conclusion

Selected: **onnx**. Measured improvement with passing validation.

Single-session measurements; small gains may be timing noise. Repeat independently.

## Baseline validation

```json
{
  "passed": true,
  "finite": true,
  "maximum_absolute_error": 0.0,
  "mean_absolute_error": 0.0,
  "maximum_relative_error": 0.0,
  "relative_error_denominator_floor": 1e-08,
  "class_agreement": 1.0,
  "accuracy_measured": true,
  "baseline_accuracy": 0.775,
  "optimized_accuracy": 0.775,
  "baseline_correct": 155,
  "optimized_correct": 155,
  "validation_examples": 200,
  "atol": 0.001,
  "rtol": 0.01,
  "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
  "accuracy_note": "Accuracy on supplied labels only."
}
```

## Candidate: onnx

```json
{
  "method": "onnx",
  "details": {
    "export_path": "results\\resnet50-cpu\\20261031T201205Z-4d2b2272\\onnx\\model.onnx",
    "opset": 17,
    "graph_optimization": "ORT_ENABLE_ALL",
    "note": "Export and session creation excluded from steady-state latency."
  },
  "setup_seconds": 3.133767600171268,
  "status": "ok",
  "validation": {
    "passed": true,
    "finite": true,
    "maximum_absolute_error": 1.6808509826660156e-05,
    "mean_absolute_error": 5.905066039122176e-07,
    "maximum_relative_error": 0.2766798418972332,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 1.0,
    "accuracy_measured": true,
    "baseline_accuracy": 0.775,
    "optimized_accuracy": 0.775,
    "baseline_correct": 155,
    "optimized_correct": 155,
    "validation_examples": 200,
    "atol": 0.001,
    "rtol": 0.01,
    "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
    "accuracy_note": "Accuracy on supplied labels only.",
    "accuracy_drop": 0.0,
    "max_accuracy_drop": 0.01
  },
  "comparison": {
    "comparable": true,
    "speedup": 1.642078135093727,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 0.31153922775787324
  },
  "objective_score": 0.6089844195769172
}
```

## Candidate: quantization

```json
{
  "method": "quantization",
  "details": {
    "quantized_linear_layers": 1,
    "engine": "onednn",
    "scope": "Only Linear layers; ResNet convolution weights remain FP32."
  },
  "setup_seconds": 0.8309399001300335,
  "status": "rejected",
  "validation": {
    "passed": false,
    "finite": true,
    "maximum_absolute_error": 0.20670309104025364,
    "mean_absolute_error": 0.02531749970172983,
    "maximum_relative_error": 49156.19686113954,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 0.985,
    "accuracy_measured": true,
    "baseline_accuracy": 0.775,
    "optimized_accuracy": 0.785,
    "baseline_correct": 155,
    "optimized_correct": 157,
    "validation_examples": 200,
    "atol": 0.001,
    "rtol": 0.01,
    "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
    "accuracy_note": "Accuracy on supplied labels only.",
    "accuracy_drop": -0.01,
    "max_accuracy_drop": 0.01
  },
  "comparison": {
    "comparable": true,
    "speedup": 0.9789470351583791,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 5.997585030185948
  }
}
```

## Limitations

- Accuracy covers only the supplied preprocessed dataset.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

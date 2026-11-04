# Kernelixx measured experiment

Experiment: inference

## Environment

```json
{
  "python": "3.12.10",
  "platform": "Windows-11-10.0.26200-SP0",
  "processor": "AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD",
  "versions": {
    "torch": "2.9.1",
    "torchvision": "0.24.1",
    "numpy": "2.2.6",
    "onnx": "1.23.2",
    "onnxruntime": "1.31.1",
    "rich": "14.2.0"
  },
  "torch_runtime_version": "2.9.1+cpu",
  "driver_visible_gpus": [
    {
      "name": "NVIDIA GeForce RTX 4060 Laptop GPU",
      "driver": "591.91",
      "total_memory_mib": 8188
    }
  ],
  "cuda_available": false,
  "torch_cuda": null,
  "gpu": null,
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
  "weights": "none",
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
  "dataset": null,
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
  "weights": "none",
  "parameters": 25557032,
  "trainable_parameters": 25557032
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ok | 159.0592 | 180.3836 | 6.1770 | N/A | 102441032 | N/A |
| onnx | ok | 90.3529 | 97.4667 | 10.8906 | N/A | 102031776 | 1.7604 |

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
  "accuracy_measured": false,
  "baseline_accuracy": null,
  "optimized_accuracy": null,
  "validation_examples": 3,
  "atol": 0.001,
  "rtol": 0.01,
  "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
  "accuracy_note": "Real task accuracy was not measured. Output similarity is not accuracy."
}
```

## Candidate: onnx

```json
{
  "method": "onnx",
  "details": {
    "export_path": "results\\20261031T162549Z-113933fc\\onnx\\model.onnx",
    "opset": 17,
    "graph_optimization": "ORT_ENABLE_ALL",
    "note": "Export and session creation excluded from steady-state latency."
  },
  "setup_seconds": 2.3480079998262227,
  "status": "ok",
  "validation": {
    "passed": true,
    "finite": true,
    "maximum_absolute_error": 3.814697265625e-05,
    "mean_absolute_error": 7.787264262636503e-06,
    "maximum_relative_error": 0.0023637747031410306,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 1.0,
    "accuracy_measured": false,
    "baseline_accuracy": null,
    "optimized_accuracy": null,
    "validation_examples": 3,
    "atol": 0.001,
    "rtol": 0.01,
    "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
    "accuracy_note": "Real task accuracy was not measured. Output similarity is not accuracy."
  },
  "comparison": {
    "comparable": true,
    "speedup": 1.760420661417253,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 0.3995039800067657
  },
  "objective_score": 0.5680460482637911
}
```

## Limitations

- Synthetic inputs measure performance and output deviation, not real accuracy.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

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
  "model": "examples.custom_model:load_model",
  "weights": "none",
  "device": "cpu",
  "input_shape": [
    8,
    16
  ],
  "seed": 42,
  "threads": 1,
  "validation_batches": 3,
  "dataset": "results/synthetic-labels.npz",
  "objective": "latency",
  "min_improvement": 0.01,
  "max_accuracy_drop": 0.01
}
```

## Model

```json
{
  "name": "examples.custom_model:load_model",
  "architecture": "Sequential",
  "weights": "none",
  "parameters": 1348,
  "trainable_parameters": 1348
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ok | 0.0307 | 0.0508 | 223995.5201 | N/A | 5392 | N/A |
| onnx | ok | 0.0377 | 0.0640 | 182982.6167 | N/A | 5392 | 0.8167 |

## Conclusion

Selected: **baseline**. No validated candidate exceeded the baseline by the required margin.

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
  "baseline_accuracy": 0.20833333333333334,
  "optimized_accuracy": 0.20833333333333334,
  "validation_examples": 24,
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
    "export_path": "results\\20261031T162257Z-b0a251f1\\onnx\\model.onnx",
    "opset": 17,
    "graph_optimization": "ORT_ENABLE_ALL",
    "note": "Export and session creation excluded from steady-state latency."
  },
  "setup_seconds": 0.46752310008741915,
  "status": "ok",
  "validation": {
    "passed": true,
    "finite": true,
    "maximum_absolute_error": 8.940696716308594e-08,
    "mean_absolute_error": 1.9591728535791237e-08,
    "maximum_relative_error": 3.388107870705426e-05,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 1.0,
    "accuracy_measured": true,
    "baseline_accuracy": 0.20833333333333334,
    "optimized_accuracy": 0.20833333333333334,
    "validation_examples": 24,
    "atol": 0.001,
    "rtol": 0.01,
    "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
    "accuracy_note": "Accuracy on supplied labels only.",
    "accuracy_drop": 0.0,
    "max_accuracy_drop": 0.01
  },
  "comparison": {
    "comparable": true,
    "speedup": 0.8167330677290836,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 0.0
  },
  "objective_score": 1.2243902439024392
}
```

## Limitations

- Accuracy covers only the supplied preprocessed dataset.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

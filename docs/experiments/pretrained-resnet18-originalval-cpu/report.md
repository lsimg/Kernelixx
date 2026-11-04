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
  "model": "resnet18",
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
  "dataset": "results/datasets/imagenet-val-resnet18-200.npz",
  "objective": "latency",
  "min_improvement": 0.01,
  "max_accuracy_drop": 0.01,
  "task": "auto"
}
```

## Model

```json
{
  "name": "resnet18",
  "architecture": "ResNet",
  "weights": "default",
  "parameters": 11689512,
  "trainable_parameters": 11689512
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ok | 102.9442 | 123.5156 | 9.6314 | N/A | 46796608 | N/A |
| onnx | ok | 52.5614 | 61.1991 | 18.4323 | N/A | 46738848 | 1.9586 |
| quantization | rejected | 101.7013 | 135.8557 | 9.5321 | N/A | 45260620 | 1.0122 |

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
  "baseline_accuracy": 0.655,
  "optimized_accuracy": 0.655,
  "baseline_correct": 131,
  "optimized_correct": 131,
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
    "export_path": "results\\imagenette-cpu\\20261031T195950Z-ba2c3b12\\onnx\\model.onnx",
    "opset": 17,
    "graph_optimization": "ORT_ENABLE_ALL",
    "note": "Export and session creation excluded from steady-state latency."
  },
  "setup_seconds": 1.2456797000486404,
  "status": "ok",
  "validation": {
    "passed": true,
    "finite": true,
    "maximum_absolute_error": 1.811981201171875e-05,
    "mean_absolute_error": 1.7896504056989216e-06,
    "maximum_relative_error": 0.07084902364222213,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 1.0,
    "accuracy_measured": true,
    "baseline_accuracy": 0.655,
    "optimized_accuracy": 0.655,
    "baseline_correct": 131,
    "optimized_correct": 131,
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
    "speedup": 1.9585541467256833,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 0.12342774929328115
  },
  "objective_score": 0.510580726946867
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
  "setup_seconds": 0.30216860002838075,
  "status": "rejected",
  "validation": {
    "passed": false,
    "finite": true,
    "maximum_absolute_error": 0.28058575093746185,
    "mean_absolute_error": 0.04155475946476654,
    "maximum_relative_error": 939.258148006135,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 1.0,
    "accuracy_measured": true,
    "baseline_accuracy": 0.655,
    "optimized_accuracy": 0.655,
    "baseline_correct": 131,
    "optimized_correct": 131,
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
    "speedup": 1.0122220720000983,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 3.2822635349980867
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

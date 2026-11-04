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
  "device": "cuda",
  "input_shape": [
    8,
    3,
    224,
    224
  ],
  "seed": 42,
  "threads": 1,
  "validation_batches": 3,
  "dataset": "results\\datasets\\imagenet-val-resnet50-200.npz",
  "objective": "throughput",
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
| baseline | ok | 23.4121 | 32.1579 | 321.1945 | 225388032 | 102441032 | N/A |

## Conclusion

Selected: **baseline**. Baseline measured; no optimization candidates were requested.



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

## Limitations

- Accuracy covers only the supplied preprocessed dataset.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

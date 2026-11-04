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
  "device": "cuda",
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
| baseline | ok | 5.9107 | 9.0324 | 157.2959 | 109831168 | 46796608 | N/A |
| fp16 | rejected | 8.1675 | 10.8208 | 118.5364 | 64969216 | 23398384 | 0.7237 |

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

## Candidate: fp16

```json
{
  "method": "fp16",
  "details": {
    "scope": "FP16 parameters, buffers and inputs on CUDA"
  },
  "setup_seconds": 0.45681419991888106,
  "status": "rejected",
  "validation": {
    "passed": false,
    "finite": true,
    "maximum_absolute_error": 0.04754638671875,
    "mean_absolute_error": 0.0040668826701902435,
    "maximum_relative_error": 119.68045102417776,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 0.995,
    "accuracy_measured": true,
    "baseline_accuracy": 0.655,
    "optimized_accuracy": 0.66,
    "baseline_correct": 131,
    "optimized_correct": 132,
    "validation_examples": 200,
    "atol": 0.001,
    "rtol": 0.01,
    "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
    "accuracy_note": "Accuracy on supplied labels only.",
    "accuracy_drop": -0.005,
    "max_accuracy_drop": 0.01
  },
  "comparison": {
    "comparable": true,
    "speedup": 0.7236853382307927,
    "memory_reduction_percent": 40.84628509094978,
    "weight_reduction_percent": 49.99982904743865
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

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
  "weights": "none",
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
  "name": "resnet18",
  "architecture": "ResNet",
  "weights": "none",
  "parameters": 11689512,
  "trainable_parameters": 11689512
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | ok | 5.7187 | 7.3406 | 170.4297 | 109831168 | 46796608 | N/A |
| fp16 | rejected | 7.8340 | 9.0624 | 124.2799 | 64969216 | 23398384 | 0.7300 |

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

## Candidate: fp16

```json
{
  "method": "fp16",
  "details": {
    "scope": "FP16 parameters, buffers and inputs on CUDA"
  },
  "setup_seconds": 0.6277776998467743,
  "status": "rejected",
  "validation": {
    "passed": false,
    "finite": true,
    "maximum_absolute_error": 0.0020270347595214844,
    "mean_absolute_error": 0.00043706907083590826,
    "maximum_relative_error": 2.2192710673825125,
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
    "speedup": 0.7299736407094671,
    "memory_reduction_percent": 40.84628509094978,
    "weight_reduction_percent": 49.99982904743865
  }
}
```

## Limitations

- Synthetic inputs measure performance and output deviation, not real accuracy.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

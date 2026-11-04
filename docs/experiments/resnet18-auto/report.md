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
  "model": "resnet18",
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
| baseline | ok | 75.7649 | 87.2949 | 12.8845 | N/A | 46796608 | N/A |
| onnx | ok | 38.1774 | 40.0477 | 26.0567 | N/A | 46723488 | 1.9846 |
| quantization | rejected | 75.3749 | 79.7586 | 13.1660 | N/A | 45260620 | 1.0052 |
| compile | skipped | N/A | N/A | N/A | N/A | N/A | N/A |
| fp16 | skipped | N/A | N/A | N/A | N/A | N/A | N/A |

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
    "export_path": "results\\20261031T162530Z-7136aa3c\\onnx\\model.onnx",
    "opset": 17,
    "graph_optimization": "ORT_ENABLE_ALL",
    "note": "Export and session creation excluded from steady-state latency."
  },
  "setup_seconds": 0.9424511999823153,
  "status": "ok",
  "validation": {
    "passed": true,
    "finite": true,
    "maximum_absolute_error": 2.2649765014648438e-06,
    "mean_absolute_error": 3.7022649000088375e-07,
    "maximum_relative_error": 0.0022524124522844205,
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
    "speedup": 1.9845523589248597,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 0.15625064107210873
  },
  "objective_score": 0.5038919711555278
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
  "setup_seconds": 0.2398065000306815,
  "status": "rejected",
  "validation": {
    "passed": false,
    "finite": true,
    "maximum_absolute_error": 0.03981423377990723,
    "mean_absolute_error": 0.0078093475780139365,
    "maximum_relative_error": 206.78282002039026,
    "relative_error_denominator_floor": 1e-08,
    "class_agreement": 0.3333333333333333,
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
    "speedup": 1.0051747995685567,
    "memory_reduction_percent": null,
    "weight_reduction_percent": 3.2822635349980867
  }
}
```

## Candidate: compile

```json
{
  "method": "compile",
  "status": "skipped",
  "reason": "Inductor CPU on Windows requires an available MSVC C++ toolchain (cl.exe)"
}
```

## Candidate: fp16

```json
{
  "method": "fp16",
  "status": "skipped",
  "reason": "FP16 inference requires CUDA; CPU FP16 is deliberately unsupported"
}
```

## Limitations

- Synthetic inputs measure performance and output deviation, not real accuracy.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.

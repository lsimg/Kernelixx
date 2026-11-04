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
  "device": "cuda",
  "input_shape": [
    1,
    3,
    224,
    224
  ],
  "seed": 42,
  "threads": 1,
  "validation_batches": 2,
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
| baseline | ok | 82.4117 | 88.3697 | 11.9424 | N/A | 46796608 | N/A |

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
  "validation_examples": 2,
  "atol": 0.001,
  "rtol": 0.01,
  "pass_rule": "Every element: abs(candidate-reference) <= atol + rtol*abs(reference)",
  "accuracy_note": "Real task accuracy was not measured. Output similarity is not accuracy."
}
```

## Profile / potential bottlenecks

```json
{
  "bottlenecks": [
    {
      "operator": "aten::mkldnn_convolution",
      "calls": 20,
      "self_cpu_time_us": 62349.10000000005,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::max_pool2d_with_indices",
      "calls": 1,
      "self_cpu_time_us": 16892.699999999997,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 2408448
    },
    {
      "operator": "aten::native_batch_norm",
      "calls": 20,
      "self_cpu_time_us": 1318.8999999999733,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": -38400
    },
    {
      "operator": "aten::fill_",
      "calls": 162,
      "self_cpu_time_us": 1276.90000000004,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::empty",
      "calls": 160,
      "self_cpu_time_us": 877.899999999962,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 19908096
    },
    {
      "operator": "aten::_batch_norm_impl_index",
      "calls": 20,
      "self_cpu_time_us": 354.7999999999747,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::max_pool2d",
      "calls": 1,
      "self_cpu_time_us": 348.3000000000029,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": -1605632
    },
    {
      "operator": "aten::clamp_min_",
      "calls": 17,
      "self_cpu_time_us": 333.5000000000082,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::_convolution",
      "calls": 20,
      "self_cpu_time_us": 227.4999999999709,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::add_",
      "calls": 8,
      "self_cpu_time_us": 206.09999999999854,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::batch_norm",
      "calls": 20,
      "self_cpu_time_us": 181.50000000003547,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::addmm",
      "calls": 1,
      "self_cpu_time_us": 172.79999999998836,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 4000
    },
    {
      "operator": "aten::convolution",
      "calls": 20,
      "self_cpu_time_us": 158.5000000000091,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::conv2d",
      "calls": 20,
      "self_cpu_time_us": 125.69999999996116,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    },
    {
      "operator": "aten::as_strided_",
      "calls": 20,
      "self_cpu_time_us": 95.89999999999873,
      "self_device_time_us": 0,
      "self_cpu_memory_bytes": 0
    }
  ],
  "ranked_by": "self_cpu_time_us",
  "note": "One profiled inference after warmup; operator self-times are diagnostic, not latency measurements."
}
```

## Limitations

- Synthetic inputs measure performance and output deviation, not real accuracy.
- CPU peak inference memory is not measured; weight bytes are a separate metric.
- GPU peak measures PyTorch allocated tensors (weights, input, activations), not total device VRAM or reserved cache.
- Candidates are measured sequentially with only one target model resident on GPU.
- Transformation/setup/compilation excluded from steady-state timing; setup recorded separately.
- Training and inference settings are experimental, not universal optimization guarantees.
- CUDA is unavailable in this PyTorch installation; using CPU.

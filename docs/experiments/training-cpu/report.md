# Kernelixx measured experiment

Experiment: training

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
  "device": "cpu",
  "seed": 42,
  "threads": 1,
  "steps": 10,
  "warmup": 3,
  "batch_size": 8,
  "input_shape": [
    8,
    3,
    32,
    32
  ],
  "optimizer": "SGD lr=0.01"
}
```

## Measurements

| Method | Status | Median ms | p95 ms | Samples/s | Peak GPU bytes | Weight bytes | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| fp32 | ok | 4.4279 | 5.7152 | 1694.4988 | N/A | N/A | N/A |
| amp-fp16 | skipped | N/A | N/A | N/A | N/A | N/A | N/A |

## Loss trajectories

```json
[
  {
    "method": "fp32",
    "status": "ok",
    "metrics": {
      "device": "cpu",
      "latency_median_ms": 4.4279,
      "latency_p95_ms": 5.7152199999999995,
      "throughput_samples_per_sec": 1694.498809614586,
      "peak_gpu_memory_bytes": null
    },
    "iteration_times_ms": [
      5.0216,
      4.2595,
      5.2669,
      5.5643,
      4.1855,
      4.3262,
      4.1778,
      4.0415,
      4.5296,
      5.8387
    ],
    "losses": [
      2.259864568710327,
      2.347590208053589,
      2.328015089035034,
      2.3068747520446777,
      2.240884780883789,
      2.2836592197418213,
      2.333217144012451,
      2.342125415802002,
      2.3361666202545166,
      2.268359661102295
    ],
    "grad_scaler_scales": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  {
    "method": "amp-fp16",
    "status": "skipped",
    "reason": "AMP FP16 requires a compatible CUDA GPU"
  }
]
```

Limited synthetic CNN training experiment; no convergence or task accuracy claim.

## Limitations

- Synthetic random inputs and labels; losses do not measure real task learning.
- Same initial state and batches; optimizer and scaler reset after warmup.
- Timing includes forward, backward, optimizer and scaler; excludes data transfer.
- AMP may skip updates on overflow; scaler trajectories are recorded.
- CPU peak memory is not measured.

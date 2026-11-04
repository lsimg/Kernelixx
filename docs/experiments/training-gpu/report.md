# Kernelixx measured experiment

Experiment: training

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
  "device": "cuda",
  "seed": 42,
  "threads": 1,
  "steps": 30,
  "warmup": 5,
  "batch_size": 16,
  "input_shape": [
    16,
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
| fp32 | ok | 3.2514 | 4.0454 | 4941.0522 | 77505536 | N/A | N/A |
| amp-fp16 | ok | 5.9916 | 7.8621 | 2662.1647 | 76284928 | N/A | 0.5427 |

## Loss trajectories

```json
[
  {
    "method": "fp32",
    "status": "ok",
    "metrics": {
      "device": "cuda:0",
      "latency_median_ms": 3.2514,
      "latency_p95_ms": 4.04538,
      "throughput_samples_per_sec": 4941.052217657468,
      "peak_gpu_memory_bytes": 77505536
    },
    "iteration_times_ms": [
      2.4724,
      2.4319,
      2.2717,
      2.3909,
      2.3218,
      3.2742,
      3.547,
      2.4003,
      3.076,
      4.1159,
      3.8997,
      3.9551,
      3.794,
      3.4109,
      3.1862,
      3.447,
      3.2392,
      3.2636,
      2.4732,
      3.0457,
      4.0346,
      4.0542,
      3.9135,
      3.8007,
      3.8338,
      3.5677,
      2.7367,
      3.1281,
      3.176,
      2.8833
    ],
    "losses": [
      2.3229033946990967,
      2.303739547729492,
      2.325826406478882,
      2.3022940158843994,
      2.2904624938964844,
      2.252173662185669,
      2.3109004497528076,
      2.2674598693847656,
      2.284682512283325,
      2.2964377403259277,
      2.3311378955841064,
      2.3295626640319824,
      2.312378406524658,
      2.3085713386535645,
      2.249929428100586,
      2.306687831878662,
      2.2806038856506348,
      2.3342795372009277,
      2.319125175476074,
      2.3630685806274414,
      2.325899600982666,
      2.312350034713745,
      2.3060271739959717,
      2.319065809249878,
      2.2999067306518555,
      2.3068604469299316,
      2.2865681648254395,
      2.315066337585449,
      2.2969775199890137,
      2.30625319480896
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
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
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
    "status": "ok",
    "metrics": {
      "device": "cuda:0",
      "latency_median_ms": 5.99155,
      "latency_p95_ms": 7.862065,
      "throughput_samples_per_sec": 2662.1646504466894,
      "peak_gpu_memory_bytes": 76284928
    },
    "iteration_times_ms": [
      5.5527,
      5.6044,
      5.3887,
      6.6463,
      6.1897,
      7.0279,
      5.6768,
      4.3402,
      7.9198,
      6.7887,
      6.3689,
      6.645,
      4.6565,
      5.0343,
      6.8378,
      7.7111,
      6.5381,
      5.3647,
      4.0789,
      5.261,
      4.2155,
      4.0611,
      4.278,
      5.5908,
      6.8023,
      8.0183,
      7.7915,
      7.3502,
      5.7934,
      6.7718
    ],
    "losses": [
      2.32275390625,
      2.3038330078125,
      2.32568359375,
      2.3023681640625,
      2.2904052734375,
      2.2523193359375,
      2.3109130859375,
      2.267333984375,
      2.2847900390625,
      2.29638671875,
      2.331298828125,
      2.3294677734375,
      2.3123779296875,
      2.30859375,
      2.2498779296875,
      2.3065185546875,
      2.280517578125,
      2.334228515625,
      2.3192138671875,
      2.363037109375,
      2.3258056640625,
      2.3126220703125,
      2.3060302734375,
      2.31884765625,
      2.2998046875,
      2.306884765625,
      2.2862548828125,
      2.3150634765625,
      2.297119140625,
      2.30615234375
    ],
    "grad_scaler_scales": [
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0,
      65536.0
    ]
  }
]
```

Limited synthetic CNN training experiment; no convergence or task accuracy claim.

## Training comparison

```json
{
  "speedup": 0.5426642521551184,
  "memory_reduction_percent": 1.5748655683124335
}
```

## Limitations

- Synthetic random inputs and labels; losses do not measure real task learning.
- Same initial state and batches; optimizer and scaler reset after warmup.
- Timing includes forward, backward, optimizer and scaler; excludes data transfer.
- AMP may skip updates on overflow; scaler trajectories are recorded.
- CPU peak memory is not measured.

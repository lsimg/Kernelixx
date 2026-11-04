# GPU validation follow-up

This records the initial GPU session (45 passing tests). The subsequent
[real-image experiment](REAL_IMAGE_VALIDATION.md) adds pretrained-model accuracy
measurements and brings the suite to 53 passing tests.

Executed on 2026-11-01 (Asia/Qyzylorda); generated run directories use UTC timestamps.
Environment: Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128,
CUDA runtime 12.8, NVIDIA RTX 4060 Laptop GPU, driver 591.91.
Direct dependency snapshot: [requirements-tested-cuda.txt](../requirements-tested-cuda.txt).

## Automated checks

**45 passed, no skips**, including real CUDA FP16 inference, peak-memory measurement and AMP training.
Full output: [test-results-gpu.txt](test-results-gpu.txt). Six upstream deprecation warnings remain.
The initial GPU suite exposed an incorrect test assumption that every FP16 ResNet must pass numerical validation.
The test now verifies valid acceptance/rejection, measured memory and exclusion of invalid candidates.
An additional CUDA test uses binary fractions with exactly representable FP16 answers and requires zero deviation.
Production tolerances were not relaxed. AMP training and its CSV speedup are also exercised end-to-end.

## ResNet18 inference

This experiment was executed by the user, then its saved report was inspected and archived.
Random weights, synthetic [1,3,224,224] inputs, seed 42, one CPU thread, 5 warmup and 30 timed iterations.

| Method | Median ms | p95 ms | Peak GPU MiB | Weight MiB | Status |
|---|---:|---:|---:|---:|---|
| baseline | 5.718650 | 7.340555 | 104.743164 | 44.628723 | ok |
| fp16 | 7.834050 | 9.062415 | 61.959473 | 22.314438 | rejected |

FP16 speedup: 0.729974x; GPU peak reduction: 40.8463%.
Maximum absolute deviation: 0.00202703476; atol=0.001, rtol=0.01.
At least one output element exceeded the combined elementwise tolerance, so FP16 was rejected.
The baseline was correctly retained. Reduced memory alone does not make an invalid candidate acceptable.
[Full inference report](experiments/resnet18-gpu-fp16/report.md).

## Small CNN training

A separate measured synthetic CNN experiment: batch 16, [3,32,32] images, 30 training steps,
5 warmup steps, seed 42, SGD lr=0.01. Both variants restore the same initial state after warmup
and receive identical batches. Transfers are excluded; forward/backward/update/scaler are timed.

| Method | Median step ms | p95 ms | Samples/s | Peak GPU MiB | First loss | Last loss |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 3.251400 | 4.045380 | 4941.0522 | 73.915039 | 2.32290339 | 2.30625319 |
| amp-fp16 | 5.991550 | 7.862065 | 2662.1647 | 72.750977 | 2.32275391 | 2.30615234 |

AMP speedup: 0.542664x; GPU peak reduction: 1.5749%.
Both variants completed successfully. AMP was slower in this measured workload and saved little peak memory.
Loss trajectories and scaler scales are in the JSON; no convergence or real task accuracy is claimed.
[Full training report](experiments/training-gpu/report.md).

## Reporting corrections

- Training speedup now appears in CSV rows and Markdown/terminal tables; memory reduction is included in CSV and the comparison summary.
- Windows CSV writing uses newline="" to avoid duplicate carriage returns and spurious empty rows.
- The training report was re-rendered after these reporting fixes; its measured samples were not changed.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp results/pytest-gpu-new
.\.venv\Scripts\python.exe -m kernelixx optimize --model resnet18 --device cuda --method fp16
.\.venv\Scripts\python.exe -m kernelixx train-benchmark --device cuda --steps 30 --warmup 5 --batch-size 16 --output results/gpu-training
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/gpu-training/latest
```

The rejected single FP16 optimization intentionally exits with code 2 while retaining its report.
Use a new pytest basetemp directory; pytest owns that directory and may clear it.

## Remaining limits and next experiment

- All inputs/labels here are synthetic and inference weights are random. Numerical similarity is not real accuracy.
- Timings are individual sessions and vary with power state, thermals and background work.
- Inductor/kernel fusion remains unverified: this Windows setup lacks its required compiler/Triton stack.
- Python 3.11 and remote CI have not been executed locally; this follow-up used Python 3.12.
- The next meaningful experiment is a pretrained model on representative real images with known class labels.

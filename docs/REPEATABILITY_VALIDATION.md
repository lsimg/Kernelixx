# Repeated ResNet50 batching study

Executed on 2026-11-01 (Asia/Qyzylorda), Windows 11, Python 3.12.10, RTX 4060 Laptop GPU,
PyTorch 2.9.1+cu128 / torchvision 0.24.1+cu128, CUDA 12.8, driver 591.91.
Pretrained ResNet50 IMAGENET1K_V2, same ordered 200-image original ImageNet validation subset.

## Protocol

- Three paired trials, each with a fresh Python process and CUDA context for each batch configuration.
- Order: trial 1 batch 1 then 8; trial 2 batch 8 then 1; trial 3 batch 1 then 8.
- FP32 only, one CPU thread, seed 42, 10 warmup calls and 50 timed calls per configuration.
- Same model weights, dataset SHA-256, versions, device and measurement settings verified for all six native reports.
- Each timed sequence repeats the first fixed batch. Accuracy validation uses all 200 images.
- Cross-batch numerical equivalence checked separately before timed trials; all predictions agree and the default tolerance policy passes.
- All six processes exited successfully. No trials were dropped or selected based on their measured speed.

## Observed paired trials

| Trial | Execution order | Batch 1 images/s | Batch 8 images/s | Paired throughput ratio | Correct in both / 200 |
|---:|---|---:|---:|---:|---:|
| 1 | 1 then 8 | 70.0747 | 264.8099 | 3.778964x | 155 |
| 2 | 8 then 1 | 69.1197 | 321.1945 | 4.646931x | 155 |
| 3 | 1 then 8 | 66.8117 | 326.1665 | 4.881876x | 155 |

Batch 8 had higher throughput in **3 of 3 paired trials**.
Median paired throughput ratio: **4.646931x**; observed range **3.778964x to 4.881876x**.
Sample standard deviation of the three ratios: 0.580945.

The statistic is the median of three paired ratios. It is not calculated by pooling all 150 within-run latency samples,
and is not generally equal to the ratio of the two median throughputs.

## Descriptive throughput ranges

| Batch size | Median images/s | Minimum images/s | Maximum images/s | Sample SD images/s |
|---:|---:|---:|---:|---:|
| 1 | 69.1197 | 66.8117 | 70.0747 | 1.6776 |
| 8 | 321.1945 | 264.8099 | 326.1665 | 34.0798 |

## Resource use and quality

| Trial | Batch size | Median batch ms | p95 batch ms | Peak allocated GPU MiB | Correct / 200 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 13.712400 | 17.838525 | 158.642090 | 155 |
| 1 | 8 | 23.990450 | 45.923505 | 214.946777 | 155 |
| 2 | 8 | 23.412100 | 32.157915 | 214.946777 | 155 |
| 2 | 1 | 13.826500 | 18.668920 | 158.642090 | 155 |
| 3 | 1 | 14.702750 | 17.930125 | 158.642090 | 155 |
| 3 | 8 | 23.777300 | 29.428945 | 214.946777 | 155 |

Separate cross-batch maximum absolute error: 1.907348633e-06.
Class agreement: 100.00%; correct predictions 155/200 for both groupings.
No accuracy or numerical thresholds were relaxed. FP16 was not re-evaluated in this repeat study; it was rejected in the earlier precision experiments.

## Interpretation and limits

The observed throughput benefit occurred in all three trials, while its magnitude varied noticeably.
Three paired trials provide descriptive repeatability evidence, not a confidence interval or a statistical significance claim.
Possible sources of variation were not instrumented; no specific cause is asserted.
The ratio compares different workloads (one versus eight images per call), not individual-image response latency.
Batch formation, queueing, decoding, preprocessing, input transfers, model loading and process startup are excluded.
This remains a small ten-class image subset, not full ImageNet accuracy or production qualification.

## Reproduce or inspect

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) 'results/torch-cache'
.\.venv\Scripts\python.exe -m examples.repeat_batching --model resnet50 --trials 3 --warmup 10 --repeats 50
.\.venv\Scripts\python.exe -m examples.repeat_batching --read-summary results/batching-resnet50/repeats-20261101/summary.json
```

The first command starts new measurements in a new timestamped directory. The second only reads this saved study.
Dataset preparation and cached weights are documented in the [ResNet50 experiment](RESNET50_VALIDATION.md).
The orchestrator saves child stdout/stderr logs and completed records incrementally; failed jobs do not produce a successful summary.

## Verification and artifacts

**62 tests passed**. [Test output](test-results-repeat-study.txt). Six upstream deprecation warnings remain.
New tests cover paired-ratio aggregation, invalid/nonfinite rates, missing/duplicate pairs, and reading without launching benchmarks.
[Summary JSON](experiments/repeated-batching-resnet50/summary.json), [CSV](experiments/repeated-batching-resnet50/summary.csv),
[raw study Markdown](experiments/repeated-batching-resnet50/summary.md), [quality diagnostic](experiments/repeated-batching-resnet50/batch-equivalence.json).

- [Trial 1, batch 1 native report](experiments/repeated-batching-resnet50/trial-1-batch1/report.md).
- [Trial 1, batch 8 native report](experiments/repeated-batching-resnet50/trial-1-batch8/report.md).
- [Trial 2, batch 8 native report](experiments/repeated-batching-resnet50/trial-2-batch8/report.md).
- [Trial 2, batch 1 native report](experiments/repeated-batching-resnet50/trial-2-batch1/report.md).
- [Trial 3, batch 1 native report](experiments/repeated-batching-resnet50/trial-3-batch1/report.md).
- [Trial 3, batch 8 native report](experiments/repeated-batching-resnet50/trial-3-batch8/report.md).

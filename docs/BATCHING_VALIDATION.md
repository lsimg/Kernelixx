# ResNet50: one image versus a batch of eight

This initial study was followed by [three independent paired trials](REPEATABILITY_VALIDATION.md)
with alternating order. The observed throughput ratio varied from 3.78x to 4.88x;
all six follow-up runs retained 155/200 correct predictions.

Executed on 2026-11-01 (Asia/Qyzylorda), RTX 4060 Laptop GPU, Windows 11, Python 3.12.10,
PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, CUDA 12.8, driver 591.91.
Model: pretrained ResNet50 IMAGENET1K_V2. Same original ImageNet validation subset and model-specific transforms as the [ResNet50 experiment](RESNET50_VALIDATION.md).

## Measured results

Each row was measured after 10 warmup calls with 50 timed calls, one CPU thread, seed 42.
The first fixed batch is timed repeatedly. All 200 labeled images are used for output and accuracy validation.
Runs were sequential; one target model was resident on the GPU at a time.

| Images per call | Precision | Median batch ms | p95 batch ms | Images/s | Peak GPU MiB | Correct / 200 | Numerical check |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | fp32 | 12.971150 | 17.191565 | 73.3229 | 158.642090 | 155 | reference |
| 1 | fp16 | 19.201300 | 25.720070 | 49.5011 | 88.916016 | 155 | FAIL |
| 8 | fp32 | 23.941500 | 28.606490 | 327.3272 | 214.946777 | 155 | PASS |
| 8 | fp16 | 20.425950 | 26.267665 | 380.0936 | 127.813965 | 155 | FAIL |

FP32 throughput ratio, batch 8 / batch 1: **4.464190x**.
Peak allocated GPU memory increased by **35.4916%**.
The FP32 batch-of-eight configuration preserved class predictions and passed the cross-batch numerical check.
FP16 was faster than FP32 within batch 8, but both FP16 candidates failed numerical validation and were rejected.

## Latency and throughput have different meanings

Batch latency is the duration of the entire call: all eight submitted images complete at the end of that call.
Dividing batch time by eight is an amortized compute cost, not the response latency for an individual request.
Images/s counts individual examples, computed as batch size * repetitions / total measured time.
The throughput ratio here is a workload comparison with deliberately different batch sizes.
The ordinary precision-comparison function correctly returns comparable=false and speedup=null across these two batch sizes.
FP16 speedups inside each native report compare equal batch sizes and remain subject to validation.

## Separate cross-batch quality diagnostic

Loaded one FP32 model in eval mode and regrouped the exact same ordered input tensors into batches of 1 and 8.
Concatenated all predictions in original order, then applied the normal numerical and labeled-accuracy checks.
Maximum absolute error: 1.907348633e-06.
Class agreement: 100.00%. Correct predictions: 155 versus 155 of 200.
Default atol=0.001, rtol=0.01 and maximum accuracy drop=0.01 were retained.
[Full diagnostic](experiments/resnet50-batch-equivalence.json). It was executed after the timed runs.

## Verification

**55 tests passed**. [Test output](test-results-batching.txt). Six upstream deprecation warnings remain.
Added a negative probe whose output depends on batch size, plus a batch-independent positive probe.
Both tests include a partial final batch and verify that all labeled examples are retained.
The terminal now explicitly prints batch size and explains that milliseconds are per batch.

## Reproduce

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) 'results/torch-cache'
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet50 --weights default --device cuda --methods fp16 --objective throughput --input-shape 1,3,224,224 --dataset results/datasets/imagenet-val-resnet50-200.npz --warmup 10 --repeats 50 --output results/batching-resnet50/batch1
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet50 --weights default --device cuda --methods fp16 --objective throughput --input-shape 8,3,224,224 --dataset results/datasets/imagenet-val-resnet50-200.npz --warmup 10 --repeats 50 --output results/batching-resnet50/batch8
.\.venv\Scripts\python.exe -m examples.validate_batching --model resnet50 --device cuda --batch-sizes 1 8 --dataset results/datasets/imagenet-val-resnet50-200.npz --output results/batching-resnet50/batch-equivalence-new.json
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/batching-resnet50/batch8/latest
```

The dataset was prepared in the [ResNet50 follow-up](RESNET50_VALIDATION.md). The diagnostic preserves existing output files; choose a new output for a new check.

## Artifacts

- [Batch 1 native report](experiments/batched-resnet50-1/report.md).
- [Batch 8 native report](experiments/batched-resnet50-8/report.md).
- [Study summary JSON](experiments/resnet50-batching-summary.json), [summary CSV](experiments/resnet50-batching-summary.csv).

## Limits and practical use

- Useful for offline folders or queued images that are already available together. Collecting a live batch introduces waiting time not measured here.
- No decoding, preprocessing, input transfers, queueing or service overhead is included in timed calls.
- Peak memory is PyTorch allocated tensor memory, not total VRAM use or reserved cache.
- Results describe this model, these shapes and this hardware in one session; independent repeats are needed for robust estimates.
- Accuracy covers this small ten-class subset only. It is not a full ImageNet benchmark or production qualification.

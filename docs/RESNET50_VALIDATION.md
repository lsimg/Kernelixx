# Pretrained ResNet50: real-image follow-up

The subsequent [batching study](BATCHING_VALIDATION.md) compares one image per call
with a batch of eight and checks output equivalence on the same images.

Executed on 2026-11-01 (Asia/Qyzylorda), Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128.
Model: torchvision ResNet50_Weights.IMAGENET1K_V2 (DEFAULT), 25,557,032 parameters.
GPU: RTX 4060 Laptop GPU, CUDA runtime 12.8, driver 591.91. CPU/ORT use one thread.

## Same raw images, model-specific preprocessing

- Reused the official Imagenette archive; no additional image download was required.
- Selected the same seeded 200 original ImageNet validation filenames, 20 per class, from the same 500 eligible images.
- Verified complete equality of the ResNet18 and ResNet50 manifests, including image hashes, order and ImageNet labels.
- Used ResNet50 V2 transforms: resize shorter side to 232, center crop to 224 and official RGB normalization.
- The ResNet18 V1 experiment uses resize 256. Its prepared tensors were not reused for ResNet50.
- Classification retains all 1000 outputs. No training or prediction-based image selection was performed.
- [ResNet50 subset manifest](experiments/original-imagenet-val-resnet50-subset.json).

## CPU measurements

Warmup 5, measured repetitions 30, batch size 1.
Accuracy uses all 200 images; timing uses the first fixed image batch.

| Method | Median ms | p95 ms | Correct / 200 | Accuracy | Speedup | Numerical validation |
|---|---:|---:|---:|---:|---:|---|
| baseline | 216.444700 | 237.667675 | 155 | 77.50% | N/A | reference |
| onnx | 131.811450 | 138.968885 | 155 | 77.50% | 1.642078 | PASS |
| quantization | 221.099500 | 241.543785 | 157 | 78.50% | 0.978947 | FAIL |

Selected: **onnx**. Measured improvement with passing validation.
[Full CPU report](experiments/pretrained-resnet50-originalval-cpu/report.md).

## GPU measurements

Warmup 10, measured repetitions 50, batch size 1.
Accuracy uses all 200 images; timing uses the first fixed image batch.

| Method | Median ms | p95 ms | Correct / 200 | Accuracy | Speedup | Numerical validation |
|---|---:|---:|---:|---:|---:|---|
| baseline | 15.365350 | 20.824680 | 155 | 77.50% | N/A | reference |
| fp16 | 28.309900 | 30.930445 | 155 | 77.50% | 0.542755 | FAIL |

Selected: **baseline**. No validated candidate exceeded the baseline by the required margin.
[Full GPU report](experiments/pretrained-resnet50-originalval-gpu/report.md).

Peak allocated GPU memory: FP32 158.642090 MiB; FP16 88.916016 MiB.

## Practical model comparison

The table uses recorded individual sessions for the same raw image manifest.
Weights and preprocessing differ between models, so this compares ready-to-use configurations, not architecture depth alone.

| Model / pretrained weights | Correct / 200 | CPU baseline ms | CPU ONNX ms | CPU ONNX speedup |
|---|---:|---:|---:|---:|
| ResNet18 / IMAGENET1K_V1 | 131 | 102.944 | 52.561 | 1.959x |
| ResNet50 / IMAGENET1K_V2 | 155 | 216.445 | 131.811 | 1.642x |

## Findings and limits

- ONNX passed numerical validation and preserved all class predictions for ResNet50 on this subset.
- FP16 saved GPU memory but was slower and failed the unchanged elementwise tolerance policy.
- Dynamic INT8 Linear quantization changed some predictions favorably on this small subset, but was slower and failed numerical validation.
- Thresholds were unchanged: atol=0.001, rtol=0.01, maximum accuracy drop=0.01.
- These measurements cover 200 resized validation images from ten classes, not all ImageNet classes or a production dataset.
- Cross-model accuracy differences include different pretrained training recipes and preprocessing.
- All speedups are within-device/model comparisons; no CPU-to-GPU speedup is computed.
- Timing excludes decoding, preprocessing, model loading and conversion. Single-session timings depend on power state and background activity.
- Existing implementation tests remain at 53 passed; no implementation changes were needed for this follow-up.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m examples.prepare_imagenette --model resnet50 --samples 200 --output results/datasets/imagenet-val-resnet50-200.npz
$env:TORCH_HOME = Join-Path (Get-Location) 'results/torch-cache'
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet50 --weights default --device cuda --methods fp16 --dataset results/datasets/imagenet-val-resnet50-200.npz --warmup 10 --repeats 50 --output results/resnet50-gpu
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet50 --weights default --device cpu --methods onnx quantization --dataset results/datasets/imagenet-val-resnet50-200.npz --warmup 5 --repeats 30 --output results/resnet50-cpu
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/resnet50-cpu/latest
```

Dataset preparation is one-time; reuse an existing NPZ. Both models use identical raw images but separate preprocessed NPZ files.
The initial image archive comes from the preparation command with --download in the [ResNet18 experiment](REAL_IMAGE_VALIDATION.md).
Checkpoint, NPZ tensors and ONNX artifacts remain in gitignored results/. Only reports and the manifest are copied into docs.

NPZ SHA-256: `cb7a0aeeb81ac1bb1a962cb421151e7b6fa3b701c5c56d78519c49f43bc62b3e`.

Pretrained checkpoint SHA-256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

[Official ResNet50 DEFAULT weights and transforms](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet50.html).

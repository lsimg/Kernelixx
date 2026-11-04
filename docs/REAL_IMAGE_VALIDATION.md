# Pretrained ResNet18: real-image validation

The [ResNet50 follow-up](RESNET50_VALIDATION.md) repeats this experiment with the
same raw image manifest and model-specific preprocessing.

Executed on 2026-11-01 (Asia/Qyzylorda), Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128.
Model: torchvision ResNet18 with ResNet18_Weights.IMAGENET1K_V1, already trained on ImageNet-1K.
GPU: RTX 4060 Laptop GPU, CUDA runtime 12.8, driver 591.91. CPU uses one PyTorch/ORT thread.

## Dataset selection and preprocessing

- Downloaded the official Imagenette 160px archive and verified MD5 e793b78cc4c9e9a4ccc0c1155377a412.
- Imagenette train/val folders mix original ImageNet training and validation filenames.
- Selected only ILSVRC2012_val_*.JPEG from both folders: 500 eligible original validation images, 50 per class.
- Seed 42 selects 20 per class without replacement, then shuffles them: 200 images across ten categories.
- Used the weights-provided resize, center crop and normalization, producing float32 [200,3,224,224].
- Mapped WNIDs to original ImageNet-1K targets. Predictions use argmax over all 1000 outputs, not a restricted ten-class head.
- No training or model-based selection of favorable examples was performed.
- [Manifest](experiments/original-imagenet-val-subset.json) records all filenames, class mappings, transforms and SHA-256 values.

An exploratory run using Imagenette's own val split was replaced after inspecting its training-style filenames.
Those exploratory measurements remain in local results; the tables below use only the original ImageNet validation subset.

## CPU: original model versus ONNX and dynamic INT8 Linear

Same 200 labeled inputs for validation; timing uses the first single-image batch, 5 warmup and 30 repetitions.

| Method | Median ms | p95 ms | Correct / 200 | Accuracy | Speedup | Numerical validation |
|---|---:|---:|---:|---:|---:|---|
| baseline | 102.944250 | 123.515615 | 131 | 65.50% | N/A | reference |
| onnx | 52.561350 | 61.199080 | 131 | 65.50% | 1.958554 | PASS |
| quantization | 101.701250 | 135.855695 | 131 | 65.50% | 1.012222 | FAIL |

Selected: **onnx**. Measured improvement with passing validation.
[Full CPU report](experiments/pretrained-resnet18-originalval-cpu/report.md).

## CUDA: original FP32 versus FP16

Same 200 labeled inputs for validation; timing uses the first single-image batch, 10 warmup and 50 repetitions.
This GPU run was repeated after the test suite and CPU benchmarks had completed.

| Method | Median ms | p95 ms | Correct / 200 | Accuracy | Speedup | Numerical validation |
|---|---:|---:|---:|---:|---:|---|
| baseline | 5.910700 | 9.032385 | 131 | 65.50% | N/A | reference |
| fp16 | 8.167500 | 10.820810 | 132 | 66.00% | 0.723685 | FAIL |

Selected: **baseline**. No validated candidate exceeded the baseline by the required margin.
[Full GPU report](experiments/pretrained-resnet18-originalval-gpu/report.md).

GPU peak: baseline 104.7432 MiB; FP16 61.9595 MiB.
FP16 max absolute error: 0.04754638672.

## Interpretation and limits

- ONNX preserved all 200 class predictions, passed the numerical policy and improved measured CPU latency.
- INT8 Linear quantization preserved the class predictions on this subset but failed the elementwise numerical policy.
- FP16 changed one class prediction favorably on this small subset, but was slower and failed numerical validation.
- Every candidate used the unchanged default atol=0.001, rtol=0.01 and maximum accuracy drop=0.01.
- Speedups are computed within each device/backend comparison. No CPU-to-GPU speedup is reported.
- This is a small, balanced ten-class subset of resized ImageNet validation images. Its accuracy is not full ImageNet accuracy or proof for production data.
- Timing measures one fixed image batch and excludes image decoding, preprocessing and model loading.
- Individual sessions vary with thermals, power state and background activity; these are not statistical significance claims.

## Verification and implementation changes

**53 tests passed**, including real CUDA execution, image preprocessing, correct label mapping, original-validation filtering and deterministic balanced sampling.
[Test output](test-results-real-images.txt). Six upstream deprecation warnings remain.
The preparer rejects existing output paths to preserve datasets. Original training filenames are excluded by default.
Terminal output now includes accuracy and correct counts on supplied labels.
Accuracy-drop decisions subtract integer correct counts first, avoiding false rejection at an exact one-percentage-point boundary.
Baseline-only benchmarks now explicitly say that no optimization candidates were requested.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m examples.prepare_imagenette --download --samples 200
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet18 --weights default --device cuda --methods fp16 --dataset results/datasets/imagenet-val-resnet18-200.npz --warmup 10 --repeats 50 --output results/imagenette-gpu
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet18 --weights default --device cpu --methods onnx quantization --dataset results/datasets/imagenet-val-resnet18-200.npz --warmup 5 --repeats 30 --output results/imagenette-cpu
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/imagenette-cpu/latest
```

Preparation is a one-time step. Reuse an existing NPZ or choose a new --output rather than overwriting it.
Raw archives, prepared tensors and ONNX binaries stay under gitignored results/. Documentation contains only reports and the manifest.

## Provenance

NPZ SHA-256: `5912aa3e355514abee7f4aea0490fbd71cd15c4922afd4815a0793ddef043c02`.

Pretrained checkpoint SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

- [Official Imagenette source](https://github.com/fastai/imagenette)
- [Torchvision ResNet18 weights and transforms](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.resnet18.html)

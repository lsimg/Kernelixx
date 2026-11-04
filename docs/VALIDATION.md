# Local validation evidence

**Update, 2026-11-04:** The renamed Kernelixx project passes all **74 tests** with CUDA available,
including GPU inference, AMP training, real-image preparation, accuracy-limit checks,
batch-equivalence probes and repeated-study aggregation/read mode.
See [GPU validation](GPU_VALIDATION.md) and the
[pretrained real-image experiment](REAL_IMAGE_VALIDATION.md) for current evidence.
The [ResNet50 follow-up](RESNET50_VALIDATION.md) repeats the experiment on the same
raw images with the heavier pretrained model and its own preprocessing.
The [batching study](BATCHING_VALIDATION.md) compares one versus eight images per call,
including a separate check of FP32 output equivalence across batch sizes.
The [repeatability follow-up](REPEATABILITY_VALIDATION.md) records all three fresh-process
paired trials, their observed range, and the limits of the descriptive statistics.
The [photo review](PREDICTION_REVIEW.md) matches every prepared image to its
prediction, reproduces 131/200 correct answers, and saves all 69 mistakes.
The [cross-model comparison](MODEL_COMPARISON.md) finds 30 corrected mistakes and
six new mistakes for ResNet50, with exact source-photo/label matching.
The [user-image follow-up](USER_IMAGES.md) records real top-5 inference on seven
explicitly supplied screenshots and explains the ImageNet class-vocabulary limit.
The [release checks](RELEASE_VALIDATION.md) verify an isolated CPU installation,
installed-wheel commands outside the checkout, source-archive tests and distribution inventories.
The [Kernelixx publication checks](PUBLICATION.md) cover the renamed package,
current command/entry points, evidence provenance and public content boundaries.
The CPU-only results below are retained as a historical session.

## Initial CPU-only session

Executed on 2026-10-31, Windows 11, Python 3.12.10, PyTorch 2.9.1+cpu.
The driver sees an RTX 4060 Laptop GPU, but the installed PyTorch is CPU-only.
Direct dependency versions are in [requirements-tested-cpu.txt](../requirements-tested-cpu.txt).

## Automated checks

`python -m pytest -q --basetemp results/pytest-release --tb=short`

**42 passed, 1 skipped** (CUDA integration), with six upstream deprecation warnings.
Full output: [test-results.txt](test-results.txt). ONNX integration was executed, not skipped.
Python 3.11 is the target and has a CI matrix entry; it was not installed for local execution.
The GitHub Actions matrix was authored but has not been run on a remote service.

## Measured inference results

All listed models use random weights. Inputs are synthetic. No real task accuracy was measured.
ResNets: batch 1, 3x224x224, seed 42, one CPU thread, 5 warmup and 30 measured repetitions.
Custom MLP: batch 8, 16 features, 3 warmup and 20 repetitions, synthetic labels.
Values below come directly from the archived JSON; they are individual sessions, not general speedup claims.

| Experiment | Baseline median ms | ONNX median ms | Speedup | Max absolute error | Validation |
|---|---:|---:|---:|---:|---|
| [resnet18-auto](experiments/resnet18-auto/report.md) | 75.764950 | 38.177350 | 1.984552 | 2.2649765e-06 | PASS |
| [resnet50-onnx](experiments/resnet50-onnx/report.md) | 159.059200 | 90.352950 | 1.760421 | 3.8146973e-05 | PASS |
| [custom-onnx-slower](experiments/custom-onnx-slower/report.md) | 0.030750 | 0.037650 | 0.816733 | 8.9406967e-08 | PASS |

ResNet18 INT8 Linear quantization was numerically rejected at the default thresholds.
The custom MLP ONNX candidate passed numerical validation but was slower; the baseline won.
The MLP labels are generated random labels; their accuracy metric exercises the contract only.

## Other checks

- [PyTorch Profiler and explicit CUDA-to-CPU fallback](experiments/resnet18-profile/report.md).
- [Measured FP32 CNN training and explicit AMP skip](experiments/training-cpu/report.md).
- Editable wheel build/install, console entry point, module entry point and all command help parsers checked.
- JSON/Markdown/CSV creation and compare round trip covered by tests and CLI execution.
- Custom factories, partial dataset batches, dynamic ONNX batches and accuracy-drop rejection covered by tests.

## Unverified paths and environment limitations

- In this initial session, FP16 inference and AMP GPU training were not measured because PyTorch was CPU-only. They were subsequently exercised in the GPU follow-up linked above.
- The initial CUDA download was interrupted; the user subsequently installed the official CUDA build successfully.
- Inductor was skipped because this Windows session has no available MSVC cl.exe; no compilation or kernel-fusion success is claimed.
- No real labeled dataset or pretrained weights were used in these local checks.
- Six warnings concern the deprecated PyTorch dynamic-quantization and legacy ONNX-export APIs; the PyTorch dependency is bounded below 2.10.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m kernelixx auto-optimize --model resnet18 --device cpu --methods onnx quantization compile fp16 --seed 42 --threads 1 --warmup 5 --repeats 30
.\.venv\Scripts\python.exe -m kernelixx optimize --model resnet50 --device cpu --method onnx --seed 42 --threads 1 --warmup 5 --repeats 30
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/latest
```

Archived reports retain original machine-local paths for provenance. ONNX graph binaries remain in
the corresponding gitignored local results directories; they are not duplicated into documentation.
Re-running a command creates a new run and may produce different timings.

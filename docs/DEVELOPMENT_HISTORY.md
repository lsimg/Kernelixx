# Development stages and engineering decisions

This document records the actual local development sequence before the first
public repository import on 2026-11-04. It is a development narrative backed by
the linked reports, not a reconstructed series of Git commits. Early experiment
counts describe their historical revisions; the current suite has 74 tests.

## Phase 1 — Core infrastructure

Implemented the Python package, `python -m` and installed console entry points,
argparse commands, ResNet18/50 and custom factories, and device/environment
utilities. CPU execution was essential because the initially installed PyTorch
build did not expose CUDA even though an NVIDIA GPU was present.

**Improvement:** distinguish driver-visible hardware from an actually usable
PyTorch CUDA installation. Record an explicit CPU fallback instead of claiming
GPU execution. See [initial and later validation](VALIDATION.md).

## Phase 2 — Model analysis

Implemented parameter counts, training flags, tensor and serialization sizes,
warmup, repeated forward timing, median/p95, throughput and PyTorch Profiler
operator rankings.

**Improvement:** separate weight storage from inference peak GPU allocations.
Synchronize CUDA measurements and retain raw samples. Potential bottlenecks are
profiler observations, not proof that changing an operation will be faster.

## Phase 3 — Optimization backends

Implemented explicit CUDA FP16 conversion, CPU ONNX export/Runtime graph
optimization, CPU dynamic INT8 Linear layers and guarded Inductor compilation.
Models with unsupported operators produce visible failed/skipped candidates.

**Observed limit:** Windows compilation prerequisites were absent locally;
compilation/fusion was not reported as successful. ResNet's convolutions remain
FP32 under Linear-only quantization, explaining its limited weight savings.

## Phase 4 — Automatic optimization

Added independent baseline/candidate lifecycles and four selection objectives:
latency, throughput, GPU memory and balanced. Each method starts from the same
original model and uses the same workload and device.

**Improvement:** application of a method is not success. A valid but slower
candidate stays visible and loses to the baseline. CPU memory objectives do not
substitute weight bytes for unavailable peak-memory measurements.

## Phase 5 — Output and task validation

Added finite/shape checks, elementwise absolute/relative tolerances, max/mean
deviations and class agreement. Optional labeled NPZ data enables task accuracy.

**Correction:** subtract integer correct counts before dividing by the number
of examples. This prevents rounding such as `0.98 - 0.97` from falsely exceeding
an exact one-percentage-point accuracy-loss limit. Numerical closeness, class
agreement and real accuracy remain distinct metrics.

## Phase 6 — Reports and reproducibility

Added JSON, Markdown and CSV, UTC run directories and latest report copies,
hardware/library versions, seeds, threads, settings and raw timing samples.
Saved-result comparison does not rerun inference.

**Windows correction:** CSV writers open files with `newline=""`, preventing
extra blank lines from duplicated carriage returns. Reports explicitly show
batch size, latency per batch and throughput per individual sample.

## Phase 7 — Limited training experiment

Implemented a small-CNN comparison with identical starting conditions, FP32 and
CUDA AMP, per-step timings, peak GPU allocations, training throughput, losses
and scaler behavior.

**Observed negative result:** on this small synthetic workload, AMP was slower
(about 5.99 ms versus 3.25 ms median per iteration), with about 1.57% peak allocated
memory reduction. It was retained as a measured result, not advertised as an
acceleration. No task convergence claim was made. See [GPU evidence](GPU_VALIDATION.md).

## Phase 8 — Tests, packaging and installation

Added focused unit/integration tests, CPU/CUDA backend execution, install metadata,
MIT license, wheel/sdist/source ZIP creation and inventory/source-content checks.
A fresh isolated CPU environment ran nine installed-wheel checks from outside
the checkout and source-archive tests.

**Improvements:** constrain the tested PyTorch/torchvision version pair, test
the installed console entry point, reject editable imports in wheel checks and
exclude datasets/models/environments from distribution archives. See
[historical installation checks](RELEASE_VALIDATION.md).

## Follow-up 1 — Real pretrained ResNet18 evaluation

Prepared a seeded, class-balanced 200-image subset from the official Imagenette
archive, with original ImageNet-1K label indices and all 1000 output classes.
Used pretrained weights and their official transforms.

**Data correction:** Imagenette's own train/validation folders mix original
ImageNet training and validation filenames. Default preparation now selects
`ILSVRC2012_val_*.JPEG` from both folders, avoiding that known training-overlap
source. Exploratory results were preserved locally but excluded from the final
held-out subset table. Per-image and NPZ hashes make the selection inspectable.

**Result:** CPU ONNX median 52.561 ms versus baseline 102.944 ms (1.959x), with
131/200 correct and passing numerical validation. GPU FP16 saved memory but
was slower and failed the unchanged numerical policy. See [real-image report](REAL_IMAGE_VALIDATION.md).

## Follow-up 2 — ResNet50 and honest model comparisons

Repeated the procedure on the same source photographs with ResNet50's own
pretrained weights and transforms. CPU ONNX median 131.811 ms versus baseline
216.445 ms (1.642x), retaining 155/200 correct answers and numerical validation.

**Observed limits:** FP16 and Linear INT8 did not become accepted winners simply
because some class predictions agreed or improved. ResNet18 V1 and ResNet50 V2
use different weight recipes and resize settings, so this comparison does not
isolate architecture depth. See [ResNet50 report](RESNET50_VALIDATION.md).

## Follow-up 3 — Batch size and repeated paired trials

Compared CUDA FP32 ResNet50 with batches of one and eight. Added a separate
cross-batch output-equivalence check, keeping all 200 examples and partial batches.
Then repeated three fresh-process pairs in alternating execution order.

**Improvement:** cross-batch throughput is explicitly a workload ratio, not a
single-image latency speedup. Do not mix it with same-workload optimization
speedup. Use the median of paired ratios rather than pooling repeated latency
samples or taking a ratio of unrelated medians.

**Results:** single-session throughput ratio 4.46x, with increased batch latency
and peak GPU memory. Repeated paired ratios ranged 3.779x–4.882x, median 4.647x.
All six runs had 155/200 correct. These are descriptive observations, with no
statistical significance claim. See [batching](BATCHING_VALIDATION.md) and
[three-pair repeatability evidence](REPEATABILITY_VALIDATION.md).

## Follow-up 4 — Inspectable individual predictions

Added CLI tools that connect the original photographs to expected and predicted
class names, CSV/JSON tables and static contact sheets. Verify source image
hashes, NPZ hashes, label order and preprocessing before matching answers.

Compared ResNet18 and ResNet50 by manifest path plus SHA-256 rather than assuming
equal row positions. **ResNet50 corrected 30 mistakes and introduced six**, for
a net gain of 24 correct answers; 125 were correct in both and 39 wrong in both.
This makes regressions visible instead of reporting only aggregate improvement.
See [photo review](PREDICTION_REVIEW.md) and [model comparison](MODEL_COMPARISON.md).

## Follow-up 5 — User-selected image files

Added explicit file-path inference with per-model official preprocessing,
top-5 class scores, source hashes and static report images. User originals are
preserved and remain outside the public source artifacts.

Seven apple screenshots produced the `Granny Smith` class on two images for
ResNet18 and five for ResNet50. Unexpected answers remained in the output.
**Validation limit:** the fixed ImageNet class list has no generic `apple` label;
choosing `Granny Smith` does not establish the apple's cultivar. Scores are not
calibrated confidence and these seven examples are not a representative accuracy
test. See [user-image report](USER_IMAGES.md).

## Follow-up 6 — Kernelixx public naming and documentation

Renamed the former KernelLite application to **Kernelixx**: package directory,
imports, command, Profiler marker, package metadata, report branding, examples,
tests, install tools and distribution archive names. Descriptive module filenames
retain their responsibilities. The new invocation is `python -m kernelixx`.

Expanded the English README, added a Russian usage guide, this development
history, architecture/data-flow reference, changelog and contribution guide.
Public evidence copies normalize branding and machine-specific paths; archived
metric numbers are not recomputed or changed. Original local `results/` artifacts
are retained. Source/distribution audits exclude raw images, model weights,
private data and virtual environments. Revalidation is in [PUBLICATION.md](PUBLICATION.md).

## Discussed extensions, not implemented

Gemma and Ollama were investigated as possible text-model experiments. Current
input/output contracts, metrics and optimization backends do not implement LLM
generation, KV-cache accounting, first-token latency or an Ollama adapter. These
remain explicit future work and are not listed as delivered capabilities.

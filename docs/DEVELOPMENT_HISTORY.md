# Development stages and engineering decisions

This document records the actual local development sequence before the first
public repository import on 2026-11-04. It is a development narrative backed by
the linked reports, not a reconstructed series of Git commits. Early experiment
counts describe their historical revisions; the current suite has 90 tests.

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

## Follow-up 7 — Supplied GGUF text model

Added a separate native CPU backend for a user-supplied SmolLM2-135M-Instruct Q4_K_M
file, with optional dependencies, isolated worker processes, full raw token outputs,
first-token/decode timing and sampled process RSS. Compared one versus four CPU
threads on fixed English prompts, with warmup and five repetitions each.

Median request latency fell from 522.105 to 241.760 ms; all generated token ids and
stop reasons agreed. Decode throughput rose from 58.801 to 118.565 tokens/s while
sampled working set rose slightly. These are native execution-setting observations,
not savings caused by quantization or proof of answer quality. No FP32 reference,
PyTorch transformations or GPU execution were involved. See [GGUF evidence](GGUF_SMOLLM2.md).

## Discussed extensions, not implemented

Gemma and Ollama were investigated as possible text-model experiments. The GGUF
CPU runner adds a limited native text benchmark, but does not implement an Ollama
adapter, Gemma-specific formatting, GPU GGUF, exact KV-cache allocation accounting
or logit-level LLM correctness validation. These remain explicit future work.

## V2 modernization — 2026-11-07

The preceding sections describe historical releases. Current work starts from the
verified 161-member Kernelixx source ZIP, not from a newly scaffolded project.
Archive provenance and the unchanged baseline are recorded in
[INITIAL_TECHNICAL_AUDIT.md](INITIAL_TECHNICAL_AUDIT.md). The original archives and
historical numerical evidence were preserved. The initial local suite had 90
passing tests. The first test invocation hit sandbox temporary-directory access
errors; explicit workspace temporary locations resolved those fixture errors.

| Stage | Files / delivered change | Verification and remaining boundary |
|---|---|---|
| 0. Archive/audit | INITIAL_TECHNICAL_AUDIT.md, safely extracted `.audit/original` | CRC/path checks, SHA-256, all 161 source files matched; no source archive overwritten |
| 1. Architecture | ARCHITECTURE_V2.md | Existing package/modules extended; no GUI/server or unnecessary adapter hierarchy |
| 2. Memory | core/memory.py, core/benchmark.py, optimizer.py, optional dependency extras | Scoped RSS/delta/lifetime peak, CUDA allocated/reserved/transient; real CPU/CUDA tests; NVML absent, process VRAM unavailable |
| 3. Streaming | core/data.py, validator.py, optimizer.py | NPZ spooling/NPY mmap, per-region unmapping, ordered/tail batches, chunked scalar validation; legacy list helpers retained |
| 4. Pipelines | core/pipeline.py, optimizer.py, fusion.py, quantization.py | Preflight before copying; FP16→compile and ONNX→dynamic INT8; explicit failures/skips; no exhaustive search |
| 5. Benchmarking | core/benchmark.py, core/repeat.py, reporting/reporter.py | Cold/setup/steady scopes, fresh workers, alternating pairs, source/input hashes, resource constraints and ACCEPT/REJECT/INCONCLUSIVE |
| 6. GGUF | llm/gguf.py, llm/reporting.py | Real local template rendering, prompt evaluation, token/text/logit checks, strict task checks, batch/KV/offload controls; GPU llama.cpp unavailable |
| 7. Triton | OPTIMIZATION_METHODS.md | Deferred: no demonstrated specialized-kernel advantage or compatible installed Triton; no placeholder kernel |
| 8. Model contracts | Existing loader, runner, numerical/classification validation; separate GGUF contract | ResNet/custom/GGUF compatibility retained; detection/transformer and arbitrary structured-output adapters deferred |
| 9. Export | core/export.py | Checksummed ONNX or source-state/execution recipes, native GGUF config, actual reload+complete validation; failed exports remain unverified |
| 10. CLI | cli.py | profile-memory, optimize-pipeline, benchmark-repeat, optimize-llm, export; help/config errors/exit codes/report reading tested; old commands retained |
| 11. Regression | tests/test_v2.py plus unchanged existing tests | Real CPU/CUDA/ORT/GGUF integration; focused chunk/tail/error/resource/checksum tests; see current counts in PERFORMANCE_RESULTS.md |
| 12. Measurements | tools/compare_implementations.py, preserved local results and curated evidence | Same hardware/weights/data and alternating fresh processes; negative observations retained, no invented speedup |
| 13. Documentation | READMEs, V2/memory/methodology/methods/LLM/results docs | Current behavior separated from historical evidence; limitations and unverified backends explicit |

Two measurements corrected implementation assumptions. The first ONNX INT8 chain
failed its unchanged numerical test; reduced-range per-channel weights addressed
the observed large x86 error without relaxing validation. The first full-system
streaming experiment increased RSS because touched whole-file mmap pages remained
resident. Mapping/copying/unmapping one region at a time reduced that working-set
cost. Both attempts were retained instead of presenting only successful results.
Cold-forward and steady-state scope ordering was also corrected, but the measured
GPU latency regression remained; its hardware/runtime cause is not established.

No general inference-speed improvement is inferred from lower data memory. FP16
can remain rejected numerically even when it is faster. Strict expected-answer
checks can fail while token and native-logit consistency pass. The CPU-only
llama.cpp build cannot verify offload despite a CUDA-capable PyTorch installation.
Exact KV-cache byte accounting, inference recomputation, backward checkpointing,
custom Triton kernels, generic task adapters, GPU ORT and statistical LLM selection
remain explicit future work, not completed features.

Final timing follow-up: raw GPU sequences drifted substantially even with a
larger warmup count. Added first/last-third drift diagnostics, a nullable verified
steady-state field and INCONCLUSIVE selection for unstable/insufficient timings.
Re-ran the CPU ONNX search with ten warmup calls: all three pairs passed numerical
and drift gates, median latency ratio 2.53337×; actual selected artifact reload
passed again. Final suite: 124 passed, no skips, including the real local GGUF test.
Source/wheel/sdist byte-content audits and target-installed wheel smoke checks
are separate from source tests; original archives remain unchanged.

## Supplied SoupLite reference archive — 2026-11-08

The user subsequently supplied GPU-KERNEL.zip. Audited all 35,020 archive entries
and full CRC; excluded 257 symlinks. Identified souplite-v5-memory as the primary
reference by launcher/docs and byte-identical solution files in its verified
evidence ZIP. Reconciled both saved result/checks/source digests without treating
historical T4 numbers as current Kernelixx measurements. Selected CPU chunked-loss
tests passed (9); the T4 training workload was not rerun on the 8 GB laptop.

The concrete transferable reliability pattern was source/check freshness. A new
regression reproduced export accepting a changed forward with identical weights.
Added direct model/factory and Kernelixx package source fingerprints to reports,
PyTorch recipes and independent worker comparisons. Export/reload now reject known
source changes; legacy provenance remains explicitly unverified. The existing
streaming architecture stayed in place; no SoupLite kernel, web app, Qwen hooks
or inference recomputation was transplanted. Full reconciliation and limits:
[SOUPLITE_REFERENCE_AUDIT.md](SOUPLITE_REFERENCE_AUDIT.md).

## GitHub architecture publication — 2026-11-08

Added a detailed Russian before/after migration guide with subsystem tables and
workflow/memory/worker/GGUF/export diagrams. Retained the user's subsequent CPU
ONNX experiment and successful reload in a separate normalized evidence JSON
(median three-pair ratio 2.448037, not pooled with development measurements).
Corrected CPU RSS rounding to 8.83% and clarified that CUDA-scenario RSS savings
are process RAM, not VRAM. Public log copies redact local workspace paths; raw
local evidence and original archives remain unchanged. A separate LF-normalized
source checkout passed all 127 tests before publication. Remote CI status belongs
to the published commit and is not inferred from that local result.

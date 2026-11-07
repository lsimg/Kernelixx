# Kernelixx V2 — adaptive, evidence-driven experiments

This design extends the existing CLI and runner protocol. V2 names the design,
not a claim that every proposed backend or adapter is delivered.

The [detailed before/after guide](MIGRATION_V2.ru.md) compares this architecture
against commit `0325bbf`, maps every changed module and provides workflow,
worker, memory, native-LLM and export diagrams with measured tradeoffs.

## Components and flow

`loader → analyzer/hardware → batch source → baseline → bounded pipeline planner
→ transform → numerical/quality checks → resource policy → paired benchmarks
→ conservative selector → verified export/report`.

Existing core modules keep ownership. Add only `core/memory.py` for scoped
observations, `core/data.py` for repeatable streaming inputs, `core/pipeline.py`
for compatibility/chains, `core/repeat.py` for subprocess trials, and
`core/export.py` for artifact manifests/reload verification. No new package tree.
The existing runner protocol remains Tensor-to-Tensor: ResNet classification and
custom numerical/classification models are fully supported contracts. GGUF keeps
its own token/prompt contract; formal detection/transformer adapters are deferred.

## Data and resource ownership

The batch source yields ordered CPU input/optional label pairs, retaining the
last partial batch. Synthetic inputs are generated from a fresh seeded generator
on each pass. NPY directories are read via mmap; existing NPZ members are copied
to a temporary on-disk workspace using streaming I/O, then mapped. Batch copies
are bounded by configured input batch size. Baseline outputs are spooled per batch
to disk and consumed one at a time; no complete output list is retained.
Scalar validation state and bounded element chunks preserve float64 comparison,
elementwise tolerance, sample-weighted accuracy and element-weighted mean error.

The original CPU model is immutable. One independent candidate is constructed
after preflight; incompatible chains are skipped before copying. No concurrent
GPU candidates. Temporary files, maps and samplers use context managers/finally.
Allocator cache release at experiment boundaries is housekeeping after references
are released, never a tensor-lifetime fix. Serialization size uses temporary disk,
not a retained model-sized RAM buffer. Training recomputation is a future opt-in
experiment, not an inference optimization.

## Adapters, backend compatibility and fallback

Existing transform functions are the adapters; no parallel class hierarchy.
Implemented chains: individual FP16/INT8/ONNX/compile; FP16→compile;
ONNX→dynamic MatMul/Gemm quantization. Allowed transitions explicitly encode
device/backend/precision. Runtime export/compile/probe failures stay failures.
Unknown operations cannot always be proven compatible statically: real execution
and numerical validation are required. Unsupported sequences report
`SKIPPED — INCOMPATIBLE` with a reason. Search limits cap candidates and elapsed
search budget; subprocess timeouts bound individual isolated trials.

## Metrics and selection

Memory observations distinguish scoped sampled RSS, current RSS, RSS delta,
OS lifetime peak RSS, PyTorch current/peak allocated and reserved memory, and
optional sampled NVML process VRAM. Missing sources produce null with a reason.
Loading, cold forward, post-warmup inference and end-to-end phases are separate.
Sampling/OS lifetime high-water marks are never called exact inference peaks.

Forward timings exclude transfers, validation and memory-sampler overhead. A
separate memory pass measures post-warmup inference. Cold forward includes lazy
compilation; setup excludes it. End-to-end includes preparation and validation.
Each paired trial executes baseline/candidate in distinct fresh processes,
alternating order and preserving raw records. Ratios compare identical workloads.
At least three complete pairs and improvement in every pair are required for
conservative ACCEPT; mixed/insufficient evidence is INCONCLUSIVE. This is a policy,
not a statistical significance test; no confidence interval from dependent calls.
Within-run first/last-third timing drift must also stay within 10% over at least
ten calls. Otherwise steady-state remains unverified and selection is INCONCLUSIVE.
The old single-session selector remains a descriptive compatibility API; new
pipeline selection requires repeatability confirmation and retains baseline.

## Persistence and extension

Keep schema-1 reports readable; add fields without changing old meanings.
Unique run directories own model files and raw trials, while latest holds reports.
Artifacts record source state hash, transforms, versions/hardware, benchmark and
validation, file checksums and actual reload verification. ONNX is portable within
its runtime contract; PyTorch state requires its trusted model factory; compilation
is a startup recipe, never a supposedly portable compiled binary. GGUF exports a
runtime configuration referencing hashed local weights, without copying weights.

Future backends must implement a real load/execute/validate/export contract and
declare memory sources. GPU GGUF support is conditional on native capability and
real-model evidence. Logit validation and task-quality scoring must explicitly
state availability. Triton kernels require a documented operator bottleneck,
standard-library comparison, numerical tests and device fallback first.

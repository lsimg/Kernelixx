# Changelog

## Unreleased — architecture comparison and publication, 2026-11-08

- Added a detailed before/after architecture guide with diagrams and measured costs.
- Retained a separate user-executed three-pair ONNX demo and verified export.
- Normalized public workspace paths, clarified RAM/VRAM boundaries and checked the
  LF-normalized publication source with the full 127-test suite.

## Unreleased — supplied SoupLite reference, 2026-11-08

- Audited the subsequently supplied GPU-KERNEL.zip, identified the primary
  SoupLite memory package, verified saved evidence/source hashes and ran selected
  CPU chunked-loss checks. Historical T4 figures remain separate from Kernelixx.
- Added Python source fingerprints to reject stale exports/PyTorch recipes and
  changed source across independent workers, including unchanged-weight cases.
- Preserved legacy reports with explicit unverified historical source provenance;
  no SoupLite kernels, training hooks, web application or dependencies transplanted.

## Unreleased — adaptive pipeline and memory modernization, 2026-11-07

- Audited the source ZIP and preserved the byte-identical original for comparison.
- Added scoped process RSS and distinct CUDA allocated/reserved/transient metrics,
  with optional NVML process VRAM and explicit missing-source semantics.
- Streamed NPZ/NPY datasets and reference outputs; bounded numerical scratch work
  and released consumed mmap regions on Windows. Kept legacy list APIs.
- Removed model-sized BytesIO serialization and redundant owned-model INT8 copies.
- Added compatible FP16→compile and ONNX→dynamic Linear INT8 chains, bounded search,
  isolated alternating paired trials and conservative acceptance/resource policies.
- Added separate cold/setup/steady/end-to-end timing and preserved raw observations.
- Extended native GGUF configuration, metadata templates, prompt throughput, optional
  prompt-logit and explicit task-answer checks. Local GPU offload remains unverified.
- Added checksummed ONNX/PyTorch execution recipes and local GGUF configuration
  export, with actual runtime reload validation before reporting success.
- Added CLI commands, regression/integration tests and V2 methodology/results docs.
- Retained negative results; lower data memory does not imply faster GPU inference.

## Unreleased — GGUF CPU text-model benchmark, 2026-11-04

- Added a per-prompt analysis derived from the preserved measurements, a Russian
  explanation and prioritized validation milestones with protocols/decision criteria.
- Added `gguf-benchmark` with optional llama-cpp-python/psutil dependencies and fresh
  native CPU worker processes for each thread setting.
- Added GGUF header/SHA-256 metadata, explicit ChatML/raw prompts, greedy prefix-reset
  generation, warmup and full repeated request/token records.
- Added first-token and request latency, decode/overall tokens per second, sampled
  process RSS, exact token agreement and comparison of matching output workloads.
- Extended the existing JSON/Markdown/CSV pipeline and `compare` reader for GGUF reports.
- Executed the supplied SmolLM2 Q4_K_M file on 1/4 CPU threads and retained actual
  responses, raw measurements, negative/unsupported scope and model hash.
- Added 16 focused tests; the full local suite now has 90 tests. Optional GGUF runtime
  is not required by the PyTorch-only installation or the standard CI matrix.

See [the measured experiment](docs/GGUF_SMOLLM2.md). Gemma/Ollama and GPU GGUF
remain outside this delivered backend.

## 0.1.0 — first public Kernelixx release, 2026-11-04

### Delivered

- CLI-only PyTorch analysis, profiling, benchmarking and validated automatic selection.
- ResNet18/50 and custom single-Tensor model factories; automatic CPU fallback.
- CUDA FP16, CPU ONNX Runtime, CPU dynamic INT8 Linear and guarded Inductor integration.
- Numerical deviation, class agreement and optional labeled accuracy with user policies.
- Raw repeated timings, median/p95, throughput, separate weight and CUDA peak accounting.
- JSON/Markdown/CSV, original run directories, latest report copies and read-only comparison.
- Limited FP32/AMP training experiment, real-image preparation and per-photo review tools.
- Cross-batch equivalence, paired fresh-process trials and two-model prediction comparisons.
- Explicit user-image inference with top-5 outputs and static contact sheets.
- Automated tests, Windows/Linux CPU CI matrix, source/wheel packaging and installed-wheel checks.

### Corrections and improvements made during local development

- Fixed CUDA availability reporting when the NVIDIA driver was present but PyTorch was CPU-only.
- Corrected real-image selection to exclude original ImageNet training filenames by default.
- Mapped local Imagenette folders to ImageNet-1K indices while retaining all 1000 outputs.
- Fixed floating rounding at the allowed accuracy-drop boundary using integer counts first.
- Fixed Windows CSV newline handling and made per-batch timing explicit in terminal output.
- Kept numerical rejection active even when class accuracy agreed or improved.
- Distinguished throughput ratios for changed batch sizes from same-workload latency speedups.
- Used paired-ratio aggregation and independent child processes for repeated batching experiments.
- Matched photo reviews by manifest path and SHA-256, preserving corrected and regressed cases.
- Constrained the supported torch/torchvision pair and audited installed wheels outside the checkout.

### Naming and publication

- Renamed KernelLite to Kernelixx across the package, imports, command, report branding and archives.
- Added detailed architecture, development history, Russian guide and contribution instructions.
- Normalized machine-specific paths in public evidence; numerical experiment values were retained.
- Kept raw results, private photographs, checkpoints and environments out of Git/public source archives.

This is one initial public import. The detailed local sequence and earlier
per-stage test counts are documented in [DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)
and the linked reports; no prior Git history is implied.

### Known limits

No verified kernel-fusion evidence, CPU peak-memory measurement, generic LLM/Gemma/Ollama
backend, GPU ONNX execution or universal training optimizer. The quantization API is
deprecated upstream; torch is bounded below 2.10 pending migration. Smaller memory
or successful conversion alone does not establish an accepted optimization.

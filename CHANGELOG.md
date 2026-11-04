# Changelog

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

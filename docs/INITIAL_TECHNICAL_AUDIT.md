# Initial technical audit — 2026-11-07

Follow-up, 2026-11-08: the user subsequently supplied `GPU-KERNEL.zip`. It contains
SoupLite rather than Kernelixx sources. Its integrity, version/evidence reconciliation
and applicability are recorded in [SOUPLITE_REFERENCE_AUDIT.md](SOUPLITE_REFERENCE_AUDIT.md).
The initial-state findings below retain their original chronology.

## Provenance and scope

No chat attachment was registered. The available source archive is
`dist/Kernelixx-0.1.0-source.zip`, SHA-256
`255f16329c2fc0e58125fd2e5d523a2277d421fc9305020c637dc8e7385cf37c`.
All 161 members passed CRC verification; no absolute, parent-traversal,
drive-qualified or symbolic-link entries were found. Root: `Kernelixx-0.1.0/`.
Every member matches the initial working tree byte for byte. A safely extracted,
unchanged comparison copy is in `.audit/original/`; neither archive was modified.
The 137-member KernelLite archive is older: different package name, no GGUF
implementation, and predates the rename documented in CHANGELOG. Git HEAD at
inspection was `0325bbf`. No AGENTS.md applies in this workspace.

## Architecture and capabilities

| Component | Responsibility / actual execution | Limit / coverage |
|---|---|---|
| core/analyzer.py | CPU/CUDA PyTorch Profiler, top 15 self-time operators | One diagnostic forward; not evidence for a new Triton kernel |
| core/benchmark.py | Runner protocol, synchronized forward timings, median/p95, sequential throughput, tensor/storage sizes | CUDA allocated peak only; CPU peak absent; tests execute real CPU/CUDA |
| core/optimizer.py | Seeded original and independent candidates, validation policy, normalized objective selection | No compositions, selection uses one session |
| core/validator.py | Float64 elementwise error, max/mean absolute, relative error, classification/label counts | List inputs retain all outputs; full-sized double/error temporaries |
| precision.py | CUDA FP16 model/input conversion | CUDA >= 5.3; op support established by execution |
| quantization.py | CPU dynamic INT8 Linear | Convolutions stay FP32; deprecated torch.ao API, torch < 2.10 |
| graph.py | Legacy opset-17 export, dynamic batch, CPU ORT_ENABLE_ALL | CPU only; one Tensor input/output; real ORT tests |
| fusion.py | Guarded Inductor compilation | Missing Triton/MSVC skips; fusion explicitly unconfirmed |
| models/loader.py | torchvision ResNet18/50 and trusted module:factory | CPU FP32, single input/output; no detection/transformer contracts |
| llm/gguf.py | Local llama.cpp CPU, fresh worker per thread setting, greedy generation, sampled RSS and exact token comparisons | Hardcoded ChatML/raw, no offload/KV controls, no prompt-only rate or task-quality scoring |
| training/benchmark.py | Small CNN SGD FP32/AMP forward/backward | Synthetic experiment; retains all training batches on target device |
| reporting/reporter.py | Versioned JSON, Markdown, CSV, unique run and latest copies | No reusable model export/reload contract |
| cli.py / utils/device.py | argparse, exit codes, CPU fallback, determinism, version/driver discovery | Existing CLI and report schema must be retained |

## Strengths

The implementation already separates accuracy from output similarity and storage
from allocator memory. Raw timing samples and negative results are retained.
CUDA synchronization is correct for wall-clock forward latency. Last partial
validation batches are retained. Accuracy-drop boundaries use integer counts.
Unsupported transforms and actual execution failures are visible. Optional ORT
and llama.cpp imports do not prevent basic PyTorch usage. Windows/Linux CPU CI is
defined; local results do not establish that remote CI passed.

## Confirmed code-level costs and weaknesses

1. `make_batches` materializes NPZ arrays and copies every batch. During construction
   both arrays and copies coexist. Synthetic batches are also all retained.
2. `outputs` returns complete output lists; optimizer retains the entire baseline
   list throughout candidate search and creates another complete candidate list.
3. `validate` converts two whole batches to double and allocates full error and
   relative-error tensors. Its state could instead be scalar accumulators.
4. `TorchRunner` serializes every state_dict into a BytesIO just to count bytes.
   This adds a model-sized RAM buffer; the artifact is then discarded.
5. Every candidate is deep-copied before compatibility is known. Dynamic
   quantization copies its already independent model again by default.
6. Only `max_memory_allocated` is reported. Reserved, process VRAM, RSS, model-load
   and cold-start measurements are absent. The existing report correctly says so.
7. Baseline outputs and the candidate probe run before benchmark; an added
   cold-start metric at the old benchmark location would therefore be misleading.
8. `select_best` can accept a small single-session gain. Independent paired
   experiments exist only for batch-size comparisons in examples/repeat_batching.py.

These are established allocation/lifecycle facts, not measured speedup claims.
No per-operator bottleneck justifying a custom kernel has yet been established.

## Existing evidence and environment

Initial suite: **90 passed**, 6 upstream deprecation warnings, in 10.04 seconds
with `--basetemp=results/audit-original-tests`; raw log retained there with `.txt`
suffix. An earlier attempt used an inaccessible default temp directory: 63 passed,
27 fixture setup errors; rerouting temporary files resolved this environment issue.
No tests were removed or bypassed.

Measured environment: Windows, Python 3.12.10, torch 2.9.1+cu128, RTX 4060 Laptop;
16.99 GB physical RAM (approximately 4.64 GB available at inspection). ORT, ONNX,
psutil and llama_cpp installed; Triton and pynvml absent. These are observed
capabilities, not an offload verification. Python 3.11/Linux require CI verification.

The archive preserves CPU/CUDA ResNet, batching, training and SmolLM2 reports.
For example, the historical three-pair batch-1/batch-8 median throughput ratio is
4.6469, explicitly a different-batch throughput comparison. Historical GGUF CPU
request medians are 522.105/241.760 ms for 1/4 threads, with exact greedy token
agreement; this is not task quality or an FP32 quantization comparison. Historical
evidence must not be relabeled as measurements of the new implementation.

## Risks and priorities

Priority 1: distinct, well-scoped memory metrics and streaming data/validation;
retain legacy list APIs for examples. Priority 2: separate cold/setup/steady-state,
bounded compatible chains and isolated alternating paired trials with conservative
selection. Priority 3: reload-verified artifacts and GGUF runtime configuration,
capability detection and template handling. Preserve old commands/report readers.

NPZ cannot be honestly advertised as in-place mmap; spool its NPY members to
temporary disk or accept a directory of NPY arrays. Windows mmap handles must be
closed before cleanup. CPU RSS sampling can miss spikes; lifetime OS peaks must
be named separately. WDDM can withhold per-process VRAM. Compile is lazy and
platform dependent. CPU dynamic quantization is not general CNN quantization.

## Safe modernization plan

Document V2 contracts before source edits. Add focused tests for streaming
equivalence, incomplete tails, malformed inputs, failure cleanup, memory semantics,
pipeline compatibility, repeatability, GGUF capabilities and export reload.
Run the existing regression suite after each major change. Compare the preserved
original and new implementation with fixed seeds/weights/data and fresh processes.
Record raw observations including regressions and inconclusive outcomes. Defer
Triton, inference recomputation, detection and transformer adapters until profiling
and complete task-specific tests justify them. Do not claim unsupported features.

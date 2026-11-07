# Current performance and verification — 2026-11-07–08 (Asia/Qyzylorda)

These are measured observations on the local Windows 11 / Python 3.12.10 host,
16 GB class RAM, PyTorch 2.9.1+cu128 and NVIDIA RTX 4060 Laptop GPU. CPU and CUDA
results are kept separate. No general speedup or significance claim is made.

## Source and comparison conditions

The original is the safely extracted source ZIP identified in the initial audit.
Both implementations used pretrained ResNet50, the same 200-image labeled NPZ,
batch 8, one CPU thread, seed 42, atol=1e-3, rtol=1e-2, 5 warmup and 30 timings.
Cached checkpoint SHA-256:
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
Each measurement used a fresh process; order alternated original/new, new/original,
original/new. An identical external 2 ms sampler measured whole-experiment RSS.
Table entries are medians of three process observations, except fixed size/accuracy.
The whole-experiment RAM scope includes load/validation/setup; it differs from the
engine's separate post-warmup inference memory scope. Peaks may fall between samples.

## CUDA: original implementation versus new implementation

The FP32 baseline workload is the same; the framework changes its data/resource
management, not its model mathematics. The FP16 candidate was rejected in both
versions, so baseline remains selected.

| Metric | Original | New implementation | Change (new/original − 1) |
|---|---:|---:|---:|
| Median batch latency (ms) | 36.601 | 41.990 | +14.73% |
| Throughput (samples/s) | 216.100 | 188.781 | -12.64% |
| Sampled experiment peak RAM (MiB) | 1434.172 | 1368.137 | -4.60% |
| PyTorch peak allocated (MiB) | 214.947 | 214.947 | +0.00% |
| Serialized model (bytes) | 102545599 | 102545599 | +0.00% |
| Labeled accuracy (%) | 77.500 | 77.500 | +0.00% |
| Maximum error vs own baseline | 0 | 0 | 0 |
| End-to-end experiment (s) | 8.042 | 9.405 | +16.95% |

Full process VRAM: **not measured** (pynvml absent; WDDM may also withhold it).
Allocator values above are not full process VRAM. GPU baseline latency regressed
in all three pairs. Changing cold/steady scope ordering did not eliminate the
regression; its hardware/runtime cause has not been established. Do not advertise
this modernization as a confirmed GPU latency improvement. Whole-experiment RSS
fell in every final pair, but extra validation/spooling/measurement work increased
end-to-end time. Lower data memory does not automatically imply faster inference.

FP16 had maximum absolute error 0.0475307405, mean absolute error 0.00148064372,
class agreement 1.0 and unchanged 155/200 accuracy. It still failed the elementwise
numerical policy. No threshold was relaxed to accept it. Both versions produced
the same recorded error aggregates and class counts.

The earlier retained attempt (`results/v2-original-vs-new-cuda`) increased median
RSS (1438.4 → 1492.8 MiB). It motivated per-region mapping/unmapping, rather than
retaining visited whole-file mmap pages. Final CUDA raw observations are in
`results/v2-final-cuda/summary.json`; original failed-attempt data is preserved.

## CPU: original implementation versus new implementation

The CPU experiment measured baseline only, with the same 200-image pretrained
workload and three alternating pairs. CUDA metrics are unavailable for this CPU
execution; driver presence does not change that scope.

| Metric | Original | New implementation | Change (new/original − 1) |
|---|---:|---:|---:|
| Median batch latency (ms) | 948.374 | 962.168 | +1.45% |
| Throughput (samples/s) | 8.412 | 8.246 | -1.98% |
| Sampled experiment peak RAM (MiB) | 984.254 | 897.316 | -8.83% |
| Serialized model (bytes) | 102541119 | 102541119 | +0.00% |
| Labeled accuracy (%) | 77.500 | 77.500 | +0.00% |
| Maximum error vs own baseline | 0 | 0 | 0 |
| End-to-end experiment (s) | 60.366 | 63.350 | +4.94% |
| Peak VRAM | N/A: CPU | N/A: CPU | N/A |

Whole-experiment RSS fell in all three CPU pairs. Median forward latency rose
slightly; this is not a confirmed inference speedup. Exact source/input conditions
and per-trial raw latencies are retained in `results/v2-final-cpu/summary.json`.
Three pairs describe repeatability; no significance or confidence-interval claim.


## Confirmed pipeline execution and selection

A separate CPU test used ResNet18 with random seeded weights, synthetic
`[1,3,64,64]` inputs, three validation batches and three independent paired trials
(5 warmup / 30 timed calls). This is a different workload from the pretrained
200-image implementation comparison above; it does not measure task accuracy.

ONNX passed every independent numerical check. Paired latency speedups:
**2.47914×, 2.53474×, 2.45939×**; median **2.47914×**, range 2.45939–2.53474×.
The policy selected ONNX with ACCEPT. No statistical significance claim is made.
The selected ONNX artifact was reloaded and executed on the complete validation
stream: maximum absolute error **1.25169754e-6**, within unchanged tolerances.
Local artifact: `results/v2-export-selected/manifest.json`.

A separate CUDA FP16 ResNet18 trial failed its numerical gate, so it was rejected
without claiming any completed speedup pairs. An additional custom MLP CUDA test
(`[8,16]`, synthetic, three fresh-process pairs) passed numerical checks, but
FP16 was slower in all pairs: ratios **0.88098, 0.84792, 0.86107**, median
**0.86107×**. It received REJECT and retained baseline. Negative outcomes are
part of the delivered selector validation, not hidden failures.

The CUDA diagnostic profile ranked cuDNN convolution work ahead of batch norm
on this ResNet18 workload. It did not establish an RMSNorm/LayerNorm/softmax
bottleneck or a specialized Triton advantage. No custom kernel was added.

## Native GGUF and cache checks

Local SmolLM2-135M-Instruct.Q4_K_M.gguf, SHA-256
`8a5616477424978d93e67f448e5351a856ea8c234f8ea344ad3755a06ce74662`,
llama-cpp-python 0.3.36 CPU build. A short auto-template test used context 128,
max output 16, batch 64, microbatch 32, one warmup and two repetitions of three
fixed prompts. These settings differ from the historical GGUF experiments.

| CPU threads | Request median ms | Decode tokens/s | Separate prompt-eval tokens/s | Sampled process RSS MiB |
|---:|---:|---:|---:|---:|
| 1 | 315.926 | 93.631 | 220.581 | 182.723 |
| 4 | 143.109 | 166.700 | 633.157 | 184.531 |

Token/text consistency and captured last-prompt logits agreed between these thread
settings (maximum logit error 0). The predefined strict arithmetic answer check
failed: output was `7 + 5 = 12`, while the task explicitly requested `12` only.
Agreement is therefore not evidence of task-quality success. One process per
configuration does not establish an independently confirmed LLM winner.

A further real check requested GPU layers 0 and 4. CPU ran; the GPU setting was
**SKIPPED — INCOMPATIBLE**, because `llama_supports_gpu_offload()` returned false.
GPU llama.cpp offload and native process VRAM remain **NOT VERIFIED** despite the
working PyTorch CUDA installation.

K-cache q8_0 with V-cache f16 executed on the same file/context/prompts (8-token
output limit). Cross-cache token/text outputs matched, but logit validation failed:
maximum absolute error **0.9876480103**, mean **0.1754356703**, atol=1e-3/rtol=1e-2.
The comparison withholds speedup for this numerically rejected cache variant.
No cache-quality threshold was relaxed. Exact cache bytes and other K/V formats
remain unmeasured/unverified; process RSS is not a cache allocation measurement.

Real native tests also reloaded exported GGUF configuration/weights in fresh
processes and checked the original outputs. Weights were not copied or changed.

## Reproducibility and remaining boundaries

Curated raw timing arrays, per-process observations, numerical results, templates,
GGUF responses and cache comparison are retained in
[experiments/v2-modernization.json](experiments/v2-modernization.json). Full local
worker logs, references and native logit arrays remain under `results/v2-*`.
Historical reports were not modified to look like new measurements.

No Linux/Python 3.11 execution was performed on this host; CI configuration retains
those targets. GPU ORT, compatible Triton/Inductor execution, detection/transformer
task adapters, exact KV byte accounting, training recomputation and independently
confirmed LLM selection are not delivered as verified features. Native
configuration acceptance alone never establishes hardware execution or quality.

## Final automated checks

**124 passed, no skips**, 14 upstream ONNX/torch.ao deprecation warnings, in
48.61 seconds. This includes the unchanged original 90 tests and 34 added cases,
real CUDA allocator/release checks, actual ONNX INT8 execution, fresh subprocess
pairs, resource rejection, artifact corruption/reload and real local GGUF native
loading/template/logit/config export/reload. Full log:
[test-results-v2.txt](test-results-v2.txt).

```powershell
$env:KERNELIXX_TEST_GGUF_MODEL='C:\path\to\model.gguf'
python -m pytest -q
```

In this restricted development sandbox, temporary files were routed to the
workspace with TEMP/TMP and pytest `--basetemp`; this avoids an inaccessible
sandbox default Temp path. No fixture or existing test was removed. In a normal
terminal, the standard OS temporary directory is used. Without the explicit
GGUF model variable, the native integration case is skipped by design.

The final native integration observed actual context **256** for requested
context **128**, with effective batch **64** and microbatch **32**. The runtime
can pad context allocation; reports now retain requested/effective settings
separately. Historical/current short GGUF performance tables above state the
requested context, not an exact cache-allocation size. Native EOG/end-of-turn
markers are respected in addition to the ordinary EOS token.

## Timing stability follow-up

Inspection of the raw GPU sequences found substantial first/last-segment drift.
The GPU table reports observed post-warmup medians, not verified steady-state or
intrinsic kernel speed. A one-pair diagnostic with 50 warmup calls still showed
phase-sensitive results (25.775/28.295 ms original/new); it is not an independent
three-pair performance conclusion and does not resolve the underlying cause.

The final engine explicitly marks >10% first/last-third median drift UNSTABLE,
requires at least ten timings, and leaves steady-state latency null when unverified.
Independent selection cannot accept those observations. Earlier descriptive GPU
custom-MLP ratios remain saved, but their drifting runs would be INCONCLUSIVE under
this stricter final gate, rather than evidence of a stable performance regression.

Re-ran the full CPU ONNX search with **10 warmup / 30 timed calls**. All six
independent worker runs passed the drift gate and numerical checks. Paired ratios
were **2.53337×, 2.51069×, 2.60261×**, median **2.53337×**. Selection: ACCEPT.
The selected artifact was again reloaded and numerically validated successfully
(`results/v2-stable-export/manifest.json`). The earlier 5-warmup search remains
separate evidence; results from different timing policies are not pooled.

Build verification: wheel, sdist and new source ZIP were built and audited against
current source bytes. A target-installed wheel (using the existing dependency
environment) passed five new-command help checks, real CPU memory profiling,
ONNX transformation/validation and actual artifact reload. This is not a separate
clean dependency installation or Linux validation. Original release archives in
`dist/` were not overwritten; new outputs are in `results/v2-dist/`.

## Supplied SoupLite archive follow-up — 2026-11-08

The user subsequently supplied GPU-KERNEL.zip. It contains SoupLite rather than
another Kernelixx checkout. [Reference audit](SOUPLITE_REFERENCE_AUDIT.md) records
CRC/path checks, the primary version, matching source/result/check digests and
limits of applying its Qwen training optimization to the existing inference CLI.
Its historical T4 memory figures are not new Kernelixx results.

The targeted source-freshness change was validated by a regression that first
reproduced accepting changed forward code with identical model weights. Export
and PyTorch recipe reload now reject changed recorded Python source; legacy
reports retain explicit unverified historical source provenance. No new speedup
or memory-reduction number is claimed for this metadata/reliability change.

Final Kernelixx suite after the follow-up: **127 passed, no skips**, 14 upstream
deprecation warnings, 50.50 seconds, including the real local GGUF integration.
[Full log](test-results-souplite-followup.txt). Separately, selected reviewed
SoupLite CPU chunked-loss checks: **9 passed, 2 GPU cases deselected**, 3.98 seconds
([reference log](test-results-souplite-reference.txt)); no T4 benchmark rerun.

New follow-up build outputs live in `results/v2-souplite-reference-dist/`;
previous Kernelixx release/modernization archives and GPU-KERNEL.zip are retained.

## User-executed demo and GitHub publication — 2026-11-08

The user executed the published CLI sequence in the existing Windows environment:
127 tests passed with 14 upstream warnings in 37.50 seconds. The CPU ResNet18
ONNX demo measured a preliminary 13.0003 / 5.355 ms pair (2.428×). Its independent
ratios were 2.351193, 2.511486 and 2.448037; median 2.448037×. All six worker runs
passed numerical and observed drift checks. The selected ONNX was subsequently
exported, reloaded and numerically validated; manifest `verified=true` and direct
Python source provenance `VERIFIED`. This is a separate session from the earlier
2.53337× development run; samples are not pooled. No real task accuracy was measured.

[Retained user-demo observations](experiments/manual-demo-20261108.json).
The public JSON/test-log copies normalize machine-specific workspace paths;
original local files under `results/` are unchanged. Model weights, temporary
reference arrays, full user terminal text and original ZIPs are excluded from Git.
A publication check is not a new performance experiment. GitHub Actions results
must be read from the published commit, independently of local test success.

The complete [architecture migration guide](MIGRATION_V2.ru.md) explains old/new
flows, every changed subsystem, measured benefits and remaining costs.

Pre-publication verification ran the full suite against a separate source checkout
with LF line endings, matching Git publication normalization: **127 passed, no
skips**, 14 upstream warnings, 51.13 seconds. Real local CUDA/ORT/GGUF checks were
included. [Publication log](test-results-publication-v2.txt). Normalizing public
paths/line endings does not alter numerical observations; historical source byte
hashes describe their recorded local run and are not rewritten to match a new
publication commit. This check remains separate from GitHub Actions.

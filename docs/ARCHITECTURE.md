# How Kernelixx works

Kernelixx is a local Python CLI for controlled inference experiments. A run takes
one original model and a fixed input workload, evaluates independent optimization
candidates, validates their outputs and selects a measured improvement when one
exists. It does not train ResNet, repair task mistakes or host a model service.

## Source tree and responsibilities

```text
Kernelixx/
  kernelixx/
    __main__.py             python -m kernelixx entry point
    cli.py                  argument parsing, terminal tables, exit codes
    core/
      analyzer.py           parameter/storage information and Profiler events
      benchmark.py          runner protocol, timings and comparable metrics
      optimizer.py          experiment lifecycle and AutoOptimizer selection
      validator.py          numerical error, class agreement and accuracy
    models/loader.py        built-in ResNets and trusted module:factory models
    optimizations/
      precision.py          explicit CUDA FP16 conversion
      graph.py              ONNX export and CPU ONNX Runtime execution
      quantization.py       CPU dynamic INT8 nn.Linear conversion
      fusion.py             guarded torch.compile / Inductor entry point
    reporting/reporter.py   original run directories, latest copies, reports
    training/benchmark.py   limited small-CNN FP32/AMP experiment
    utils/device.py         devices, seeds, synchronization and environment
    llm/gguf.py             optional isolated native CPU text-model experiments
    llm/reporting.py        separate GGUF metrics, terminal and report rendering
  examples/
    prepare_imagenette.py   selected real-image dataset with manifest/hashes
    make_dataset.py         synthetic labeled NPZ contract example
    custom_model.py         minimal model factory example
    validate_batching.py    cross-batch FP32 output equivalence
    repeat_batching.py      fresh-process paired throughput trials
    inspect_predictions.py  per-photo predictions, errors and contact sheets
    compare_predictions.py  match two reviews by source path and SHA-256
    predict_images.py       two-model inference on explicit user image files
  tests/                    unit tests and real backend integration checks
  tools/
    smoke_installed.py      installed-wheel checks outside the checkout
    package_source.py       archive inventory/content/SHA-256 audit
  docs/                     methodology, development history and evidence
  .github/workflows/        CPU test/build/install CI matrix
```

Modules retain descriptive filenames such as `benchmark.py` and `validator.py`.
The application name, package directory, imports, console command and generated
release archive names use Kernelixx consistently.

## Model and input contracts

`load_model` returns a CPU FP32 `torch.nn.Module` in evaluation mode and its
metadata. Built-ins are `resnet18` and `resnet50`. `--weights none` creates random
weights for functional/performance experiments; `--weights default` uses the
torchvision pretrained configuration and may download its checkpoint.

Custom models use `package.module:factory`. The factory takes no arguments and
loads any desired weights itself. Python factories are trusted local code. The
first version expects a single floating input Tensor and a single floating output
Tensor with a batch dimension, and no mutable state that changes repeated results.

Synthetic batches come from a fixed seeded generator. Labeled NPZ datasets contain
`inputs` (finite preprocessed float32 `[N, ...]`) and `labels` (int64 `[N]`). The
feature shape must match `--input-shape`. Validation keeps the final partial
batch, while timing repeats the first fixed full batch. Image decoding, resizing
and input transfers are performed outside the timed forward pass.

## Experiment lifecycle

1. Parse and validate arguments; seed Python, NumPy and PyTorch; set thread count.
2. Resolve the requested CPU/CUDA device. An unavailable CUDA request records a
   warning and falls back to CPU. Disable TF32 and request deterministic execution.
3. Load the original model and fixed batches. Measure the baseline after warmup.
   Collect baseline validation outputs on CPU, then release the target runner.
4. Start each candidate from an independent copy of the original CPU model.
   Transformation/setup and first execution are separate from steady-state timing.
5. Execute the candidate on the same inputs and validate outputs. Measure candidates
   even when a numerical policy rejects them, so the tradeoff remains inspectable.
6. Exclude failed, unsupported or invalid candidates from selection. Score eligible
   candidates against the baseline under matching device/workload conditions.
7. Write hardware, configuration, raw timings, validation, candidates, selection,
   limitations and reports. Retain the baseline when no candidate qualifies.

Only one target runner is kept on the GPU at a time. Candidates are independent;
the engine does not compose FP16, quantization and compilation into one pipeline.

## Runner and backend boundaries

The runner protocol describes device, precision, backend, weight bytes, serialized
bytes and Tensor-to-Tensor execution. `TorchRunner` wraps PyTorch evaluation.
`OnnxRunner` loads a saved graph, uses sequential CPU execution with fixed threads
and `ORT_ENABLE_ALL`, and wraps the single output back into a Tensor.

| Method | Actual work | Boundary |
|---|---|---|
| FP16 | Convert floating parameters, buffers and inputs | CUDA; unsupported operations fail validation/execution |
| ONNX | Export opset 17 with dynamic batch axis; optimize in ORT | CPU only; incompatible operators may fail export |
| Quantization | Dynamic qint8 conversion of `nn.Linear` | CPU only; ResNet convolutions remain FP32 |
| Compile | Invoke Inductor when compiler/Triton prerequisites exist | Compilation is not proof of fusion; `fusion_confirmed` stays false |

Setup or export success alone does not establish speedup, lower peak memory or
acceptable output deviation. The Windows checks intentionally expose missing
MSVC/Triton prerequisites rather than reporting an optimization that did not run.

## Timing and memory definitions

Warmup precedes repeated `perf_counter_ns` measurements. CUDA is synchronized
before and after every measured forward, accounting for asynchronous execution.
Raw durations, median and p95 are recorded. Throughput is
`batch_size * repetitions / sum(seconds)` for this sequential workload.

`speedup = baseline_median / candidate_median`. Comparison requires matching
device, input shape, batch size and measurement settings. CPU and CUDA numbers
are not combined into a single optimization speedup. Changing the batch size is
a separate workload experiment with an explicitly labeled throughput ratio.

Weight tensor bytes, serialized artifact bytes and inference peak allocations
are different measurements. CUDA peak is `torch.cuda.max_memory_allocated` for
the target model and inputs, including outputs. It excludes CUDA context, reserved
allocator cache and other processes. Resident and transient values are separate.
PyTorch CPU peak is not measured; no RSS or weight-size proxy silently replaces it.

## Correctness and selection

Every output element must be finite, have the reference shape and satisfy
`abs(candidate - reference) <= atol + rtol * abs(reference)`. Reports include
maximum/mean absolute error, maximum relative error with a denominator floor,
and class agreement when classification applies. Numerical similarity is not
real task accuracy.

With labels, accuracy is measured over every supplied example. The permitted
accuracy drop uses integer correct-count differences divided by sample count,
avoiding floating rounding at a policy boundary. Passing labeled accuracy alone
does not waive the elementwise numerical policy.

Normalized scores are lower-is-better: latency ratio, inverse throughput ratio,
GPU peak-memory ratio, or the geometric mean of latency and memory ratios.
A candidate must pass validation and exceed the required improvement margin.
Memory/balanced objectives explicitly cannot select from unmeasured CPU peaks.

## Artifacts and failures

Each original run has a unique UTC timestamp/id directory, with `report.json`,
`report.md`, `metrics.csv`, and an ONNX file when applicable. `latest/` contains
report copies; portable ONNX artifacts stay in the original directory. `compare`
reads reports and does not launch another benchmark.

Candidate failures, CUDA OOM, export errors and unsupported methods become visible
statuses/reasons when handled by the experiment lifecycle. Invalid arguments or
fatal command errors exit 2. A single skipped/rejected optimization also exits 2
after retaining its report. Completed auto-optimization can exit 0 while retaining
the baseline; inspect `selection.improved` rather than treating exit 0 as speedup.

Raw datasets, checkpoints, environments and private images remain local under
ignored directories. Public evidence copies normalize machine-specific paths
and branding; numerical values and recorded metric meanings remain unchanged.

## Optional GGUF CPU boundary

`gguf-benchmark` uses optional llama-cpp-python/psutil dependencies without changing
the PyTorch runner contract. The parent validates the GGUF header/hash and prompt
configuration. Each requested CPU thread count runs in a fresh native worker,
with `n_gpu_layers=0`, CPU K/V operations and explicit thread counts for generation
and prompt processing. No HTTP server or hosted service is started.

ChatML formatting is explicit and checked against the GGUF's declared template;
other models require fully formatted raw prompts. The model's evaluated prefix is
reset before each warmup/timed request. Greedy generation yields token ids, allowing
timing to the first non-EOS token and each subsequent token. Tokenization, loading
and detokenization stay outside measured requests. EOS processing is included in
request latency but not counted as output text. Decode tokens/s uses N-1 output-token
intervals; zero/one-token responses have no decode-rate measurement.

The report stores every request, token sequence and answer. Reference repeats and
candidate outputs are compared by prompt ids, input/output token ids and stop reason;
speedup is omitted when agreement fails. This is not floating-logit validation or
task accuracy. No savings relative to an absent unquantized baseline are inferred.

A 10 ms sampler observes the isolated process RSS through load, warmup and requests.
It includes the process/runtime and file mapping, can miss transient peaks and is
not comparable to the PyTorch CUDA allocator metric. GGUF reports use experiment
type `gguf` under schema 1; the common saver writes JSON/Markdown/CSV and `compare`
dispatches to the matching report renderer. Failed/timed-out native workers remain
visible and cause `gguf-benchmark` to exit 2 after report saving.

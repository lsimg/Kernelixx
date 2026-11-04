# Supplied SmolLM2 GGUF: measured CPU execution

Executed locally on 2026-11-04 (Asia/Qyzylorda), Windows 11, Python 3.12.10.
The user supplied an existing file named `SmolLM2-135M-Instruct.Q4_K_M.gguf`.
Kernelixx loaded and executed that exact file using the native CPU backend.
The original file remains local; no weights or private download paths are published.

## Model identity and runtime

| Property | Observed value |
|---|---|
| File bytes | 105,453,984 |
| File MiB | 100.569 |
| SHA-256 | `8a5616477424978d93e67f448e5351a856ea8c234f8ea344ad3755a06ce74662` |
| GGUF version | 3 |
| Tensor count / metadata entries | 272 / 30 |
| Declared architecture | llama |
| File type | 15 (`Q4_K_M` in the installed runtime) |
| llama-cpp-python | 0.3.36, prebuilt Windows CPU wheel |
| psutil | 7.1.3 |
| Device | CPU; `n_gpu_layers=0`, `offload_kqv=False` |

The filename identifies the supplied artifact; its converter/source download was
not independently authenticated against an upstream artifact. Its hash fixes the
identity of this experiment. The [upstream model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
describes the original model family and its primarily English use. Kernelixx does
not claim that loading an already-quantized file caused its compact size.

## Protocol

- One fresh child process per setting, first one CPU thread, then four.
- Both prompt-evaluation and generation thread counts set explicitly.
- Context 512, maximum 64 output tokens, seed 42, zero-temperature greedy sampling.
- Three public English prompts, identically rendered in explicit ChatML; metadata
  confirms the role markers and stop-token format. All 1000-class image logic is
  outside this text-model backend.
- One warmup request per prompt, then five repetitions per prompt/setting:
  **15 measured requests and 325 non-EOS output tokens per configuration**.
- Reset the evaluated prefix before every request; repeated prompts do not reuse it.
- Tokenization/model loading/warmup/output decoding are outside request timing.
- TTFT is measured to the first non-EOS token. Request time includes EOS processing.
- Decode throughput is `sum(N-1) / sum(last_non_EOS_time - first_non_EOS_time)` for
  requests with at least two output tokens. It excludes prefill and EOS evaluation.
- Model loading occurs separately. Sampled process RSS includes the isolated
  worker's runtime, mapping, load, warmup and timed requests, at a 10 ms interval.
- Other benchmark/test workloads were not run concurrently with this experiment.

## Actual measurements

| CPU threads | Request median ms | Request p95 ms | TTFT median ms | Decode tokens/s | Overall tokens/s | Sampled process RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 522.105100 | 892.082280 | 238.313300 | 58.801044 | 35.711141 | 194.136719 |
| 4 | 241.759800 | 418.979070 | 89.561000 | 118.565476 | 78.979866 | 195.820312 |

Four threads produced a **2.160x** baseline/candidate median request-latency ratio.
Decode throughput ratio is approximately **2.016x**. Working set increased slightly.
All reference repeats and all four-thread outputs had **identical input/generated
token ids and EOS finish reasons**. A mismatch would withhold comparison ratios.

This is a composite workload of the same three prompts with equal repetitions.
It is a single sequential setting comparison, not an independently interleaved
repeat study or a statistical significance claim. Thread order, OS file cache,
thermals and background activity can affect other runs. These are not GPU results.

## Real answers, retained without quality filtering

All repeats/settings produced these same answers:

**Prompt:** In one short sentence, explain what a neural network does.

> A neural network is a type of machine learning algorithm that uses a network of interconnected nodes (neurons) to process and analyze data, allowing it to learn and adapt in complex and dynamic environments.

**Prompt:** What is 7 plus 5? Answer with the number only.

> 7 + 5 equals 12.

**Prompt:** Name three fruits, separated by commas.

```text
Three fruits separated by commas are:

- Banana
- Apple
- Orange
```

The arithmetic answer contains the right sum but violates the number-only format.
The fruit answer uses a list rather than commas. Equal tokens therefore do not
establish instruction-following quality. No real task accuracy, perplexity,
numeric logit error or unquantized reference comparison was measured.

## Reproduce and read

Install the optional native dependencies using the prebuilt CPU instructions in
the README, then supply an existing copy of the same GGUF file:

```powershell
.\.venv\Scripts\python.exe -m kernelixx gguf-benchmark --model-file "C:\Models\SmolLM2-135M-Instruct.Q4_K_M.gguf" --threads 1 4 --repeats 5 --max-tokens 64 --context 512 --output results/smollm2-gguf-cpu
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir results/smollm2-gguf-cpu/latest
```

The published copies can be read without loading a model or native library:

```powershell
.\.venv\Scripts\python.exe -m kernelixx compare --run-dir docs/experiments/smollm2-135m-q4-cpu
```

- [Full JSON](experiments/smollm2-135m-q4-cpu/report.json): per-request timings,
  tokens, answers, model/runtime metadata and validation.
- [Generated Markdown](experiments/smollm2-135m-q4-cpu/report.md).
- [Metrics CSV](experiments/smollm2-135m-q4-cpu/metrics.csv).
- [Test output](test-results-gguf.txt).
- [Supplementary tested dependencies](../requirements-tested-gguf.txt).

Published report copies retain the measured numbers/timestamp but replace the
private model location with its basename and use a relative run directory.
Derived aggregate values were checked against the raw samples, and the original
GGUF hash was verified after execution. Original full worker files/logs remain
under local ignored `results/smollm2-gguf-cpu/`.

## Coverage and limitations

The full local suite passes **90 tests**. Sixteen added cases cover workload/header
validation, prompt formatting, first-token/decode interval accounting, EOS/length
limits, empty outputs, token mismatches and repeat variability, native-worker errors,
missing dependencies, CLI/report saving and the existing saved-report reader.
Unit probes are not the source of the performance numbers above; those came from
the real supplied model. Standard CI runs these tests without downloading a GGUF
or installing the optional native backend. Native runtime execution was local.

Sampled RSS is the Windows process working set, not an exact allocation peak,
model-only storage, CUDA allocator measurement or a KV-cache size measurement.
Only CPU native execution settings were changed. PyTorch FP16/ONNX/INT8/compile
transformations, GPU GGUF, Ollama, re-quantization and fine-tuning are not part of
this experiment. Do not compare its RSS or timing directly with the earlier
PyTorch/CUDA results as if they were the same workload and memory instrument.

## Primary technical references

- [llama-cpp-python installation and native backends](https://llama-cpp-python.readthedocs.io/en/latest/).
- [Llama generate/tokenize API](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/).
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

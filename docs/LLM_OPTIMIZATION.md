# Local GGUF runtime experiments

`llama-cpp-python` remains optional (`.[gguf]`). No Ollama integration or server is introduced by this project.
Existing `gguf-benchmark` is retained; `optimize-llm` exposes the same real runner.

```bash
python -m kernelixx optimize-llm --model-file model.gguf --threads 1 4 --prompt-format auto --n-ctx 512 --n-batch 256 --n-ubatch 64 --validate-logits
python -m kernelixx optimize-llm --model-file model.gguf --threads 4 --n-gpu-layers 0 8 --prompt-format auto
```

The second command requests an offload comparison, not proof of GPU execution.
The installed native build must expose the requested constructor parameters and
return true from `llama_supports_gpu_offload()`. The inspected local build reports
false. GPU configurations are skipped as incompatible and remain NOT VERIFIED.
`n_gpu_layers`, `n_threads`, `n_batch`, `n_ubatch`, `n_ctx`, K/V types and flash
attention are recorded. Search is capped at sixteen thread/layer configurations.
Every configuration runs in a fresh process with a timeout and raw native logs.
Requested constructor settings and effective context/batch settings are separate.
The installed backend padded a requested context of 128 to 256 in the final native
integration test. A request is not an exact cache-allocation size guarantee.

`--prompt-format auto` renders the GGUF metadata chat template with the installed
backend's sandboxed Jinja formatter, BOS/EOS metadata and a user message. Unsupported
templates fail explicitly; no hardcoded ChatML fallback is used. The old explicit
ChatML option remains the default for command compatibility and checks declared
markers. Raw prompts are caller-formatted. Each run retains actual rendered
prompts, since template changes affect the workload and quality.

* TTFT measures time to first generated token excluding EOS/end-of-turn markers, including prompt evaluation
  and first-token sampling. It is not a pure prompt-throughput measurement.
* Prompt throughput comes from a separate reset `model.eval(prompt_tokens)` pass,
  before warmup. It includes native evaluation and Python dispatch, not sampling.
* Decode rate uses N−1 intervals between generated non-EOG tokens. EOS-only and
  single-token responses do not have a decode rate.
* Total generation time sums measured requests; loading/tokenization/detokenization
  are outside these request durations. Tokens and actual responses are retained.
* Sampled process RSS covers loading and a separate inference scope. It is neither
  model-only RAM nor exact KV-cache size. Native llama.cpp memory is not a PyTorch
  allocation. Process VRAM is unavailable in the current native experiment.

`--type-k/--type-v` expose f16/f32/q8_0/q4_0 only when constructor arguments and
GGML constants exist. Quantized V requires explicit flash-attention intent and
actual backend execution still decides support. Cache context/storage choices
are quality-affecting and retained in JSON. There is no stable public exact cache
allocation API in the inspected binding: `exact_cache_bytes=null`; RSS changes
must not be relabeled KV-cache bytes. Changing only constructor names is not
evidence that a cache format works on every model/device.

Validation distinguishes exact token/text consistency from model quality.
Optional `--validate-logits` captures the last prompt-token vocabulary logits via
the installed native context API, stores arrays on disk and compares in chunks
against the first configuration with atol=1e-3, rtol=1e-2. A binding that lacks
this API fails explicitly. This uses a checked private runtime accessor and may
need adaptation for future binding versions. It compares the same quantized
weights across runtime settings, not against nonexistent FP32 weights.

Prompt JSON can include `expected_exact`. Its stripped exact answer is evaluated
per request and reported separately; it does not turn token agreement into task
accuracy. The built-in arithmetic task expects `12`. A model can pass consistency
while failing that task. General task benchmarks and semantic evaluation require
a larger predefined labeled suite and are not implied by this small check.

GGUF version/header, file SHA-256, size, metadata, quantization metadata and runtime
settings are retained. Different weight files need independent reports/hashes;
no automatic cross-file quantization-savings claim is made. Runtime comparisons
are descriptive; independent paired LLM significance/selection is future work.
Historical SmolLM2 evidence stays in [GGUF_SMOLLM2.md](GGUF_SMOLLM2.md); current
verification is listed in [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md).

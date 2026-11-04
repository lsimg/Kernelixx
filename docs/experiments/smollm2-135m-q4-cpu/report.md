# Kernelixx GGUF CPU experiment

## Model and environment

```json
{
  "model": {
    "filename": "SmolLM2-135M-Instruct.Q4_K_M.gguf",
    "file_bytes": 105453984,
    "sha256": "8a5616477424978d93e67f448e5351a856ea8c234f8ea344ad3755a06ce74662",
    "gguf_version": 3,
    "tensor_count": 272,
    "metadata_entry_count": 30
  },
  "hardware": {
    "python": "3.12.10",
    "platform": "Windows-11-10.0.26200-SP0",
    "processor": "AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD",
    "device": "cpu"
  },
  "config": {
    "model_file": "SmolLM2-135M-Instruct.Q4_K_M.gguf",
    "threads": [
      1,
      4
    ],
    "context": 512,
    "max_tokens": 64,
    "warmup": 1,
    "repeats": 5,
    "seed": 42,
    "prompt_format": "chatml",
    "prompt_file": null,
    "timeout_seconds": 600
  }
}
```

## Measurements

| Threads | Status | Request median ms | Request p95 ms | TTFT median ms | Decode tokens/s | Sampled process RSS MiB | Latency ratio | Token agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | ok | 522.105 | 892.082 | 238.313 | 58.801 | 194.137 | 1.000 | True |
| 4 | ok | 241.760 | 418.979 | 89.561 | 118.565 | 195.820 | 2.160 | True |

These are CPU execution settings for one already-quantized model; no model transformation occurred.
A ratio above one indicates a lower median request latency on this workload only.
Validation checks greedy token identity on these prompts, not answer correctness.

## Inputs

```json
[
  {
    "id": "explanation",
    "text": "In one short sentence, explain what a neural network does.",
    "rendered": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nIn one short sentence, explain what a neural network does.<|im_end|>\n<|im_start|>assistant\n"
  },
  {
    "id": "arithmetic",
    "text": "What is 7 plus 5? Answer with the number only.",
    "rendered": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 7 plus 5? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n"
  },
  {
    "id": "list",
    "text": "Name three fruits, separated by commas.",
    "rendered": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nName three fruits, separated by commas.<|im_end|>\n<|im_start|>assistant\n"
  }
]
```

## Threads: 1

```json
{
  "threads": 1,
  "status": "ok",
  "device": "cpu",
  "backend": "llama-cpp-python",
  "runtime_version": "0.3.36",
  "system_info": "CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | ",
  "model_metadata": {
    "general.name": "Model",
    "general.architecture": "llama",
    "general.type": "model",
    "llama.feed_forward_length": "1536",
    "general.size_label": "135M",
    "llama.block_count": "30",
    "llama.context_length": "8192",
    "llama.embedding_length": "576",
    "llama.attention.head_count": "9",
    "general.file_type": "15",
    "llama.attention.head_count_kv": "3",
    "llama.rope.freq_base": "100000.000000",
    "llama.attention.layer_norm_rms_epsilon": "0.000010",
    "llama.attention.key_length": "64",
    "llama.attention.value_length": "64",
    "llama.vocab_size": "49152",
    "llama.rope.dimension_count": "64",
    "general.quantization_version": "2",
    "tokenizer.chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  },
  "setup_seconds": 0.2720343000255525,
  "metrics": {
    "request_latency_median_ms": 522.1051,
    "request_latency_p95_ms": 892.08228,
    "time_to_first_token_median_ms": 238.3133,
    "time_to_first_token_p95_ms": 277.51383999999996,
    "decode_tokens_per_second": 58.801044427944746,
    "overall_tokens_per_second": 35.71114077931642,
    "requests": 15,
    "generated_tokens": 325,
    "sampled_process_rss_peak_bytes": 203567104,
    "process_rss_before_load_bytes": 41537536,
    "peak_gpu_memory_bytes": null
  },
  "validation": {
    "passed": true,
    "criterion": "Exact prompt/generated token ids and finish reason",
    "different_samples": [],
    "accuracy_measured": false,
    "note": "Agreement on these prompts is not semantic quality, accuracy or a logit-tolerance test."
  },
  "comparison": {
    "comparable": true,
    "request_latency_speedup": 1.0,
    "decode_throughput_ratio": 1.0
  }
}
```

### Response: explanation

```text
A neural network is a type of machine learning algorithm that uses a network of interconnected nodes (neurons) to process and analyze data, allowing it to learn and adapt in complex and dynamic environments.
```

### Response: arithmetic

```text
7 + 5 equals 12.
```

### Response: list

```text
Three fruits separated by commas are:

- Banana
- Apple
- Orange
```

## Threads: 4

```json
{
  "threads": 4,
  "status": "ok",
  "device": "cpu",
  "backend": "llama-cpp-python",
  "runtime_version": "0.3.36",
  "system_info": "CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | ",
  "model_metadata": {
    "general.name": "Model",
    "general.architecture": "llama",
    "general.type": "model",
    "llama.feed_forward_length": "1536",
    "general.size_label": "135M",
    "llama.block_count": "30",
    "llama.context_length": "8192",
    "llama.embedding_length": "576",
    "llama.attention.head_count": "9",
    "general.file_type": "15",
    "llama.attention.head_count_kv": "3",
    "llama.rope.freq_base": "100000.000000",
    "llama.attention.layer_norm_rms_epsilon": "0.000010",
    "llama.attention.key_length": "64",
    "llama.attention.value_length": "64",
    "llama.vocab_size": "49152",
    "llama.rope.dimension_count": "64",
    "general.quantization_version": "2",
    "tokenizer.chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  },
  "setup_seconds": 0.20562619995325804,
  "metrics": {
    "request_latency_median_ms": 241.75979999999998,
    "request_latency_p95_ms": 418.97907,
    "time_to_first_token_median_ms": 89.561,
    "time_to_first_token_p95_ms": 111.22512999999998,
    "decode_tokens_per_second": 118.56547554642525,
    "overall_tokens_per_second": 78.97986591794725,
    "requests": 15,
    "generated_tokens": 325,
    "sampled_process_rss_peak_bytes": 205332480,
    "process_rss_before_load_bytes": 41406464,
    "peak_gpu_memory_bytes": null
  },
  "validation": {
    "passed": true,
    "criterion": "Exact prompt/generated token ids and finish reason",
    "different_samples": [],
    "accuracy_measured": false,
    "note": "Agreement on these prompts is not semantic quality, accuracy or a logit-tolerance test."
  },
  "comparison": {
    "comparable": true,
    "request_latency_speedup": 2.1596026303794096,
    "decode_throughput_ratio": 2.016383836374136
  }
}
```

### Response: explanation

```text
A neural network is a type of machine learning algorithm that uses a network of interconnected nodes (neurons) to process and analyze data, allowing it to learn and adapt in complex and dynamic environments.
```

### Response: arithmetic

```text
7 + 5 equals 12.
```

### Response: list

```text
Three fruits separated by commas are:

- Banana
- Apple
- Orange
```

## Limitations

- Already-quantized GGUF: no FP32 reference, quantization savings or numerical logit validation are measured.
- CPU-only native backend. PyTorch/CUDA optimizations and GPU allocator measurements do not apply.
- Sampled process RSS at 10 ms is a worker working-set observation, not exact peak/model-only memory.
- Tokenization, model loading, output detokenization and warmup are outside request timing.
- TTFT is to the first non-EOS generated token. Decode rate uses intervals between non-EOS tokens (N-1).
- Aggregate metrics mix the same fixed prompts with equal repetitions; short/EOS-only answers affect token counts.
- Each thread setting uses a fresh process, in listed order. One-session results are descriptive, not significance claims.
- Greedy token agreement on the supplied prompts is not real task accuracy or proof of general answer quality.

from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROMPTS = (
    {"id": "explanation", "text": "In one short sentence, explain what a neural network does."},
    {"id": "arithmetic", "text": "What is 7 plus 5? Answer with the number only.", "expected_exact": "12"},
    {"id": "list", "text": "Name three fruits, separated by commas."},
)


@dataclass(frozen=True)
class GGUFConfig:
    model_file: str
    threads: tuple[int, ...] = (1, 4)
    context: int = 512
    max_tokens: int = 64
    warmup: int = 1
    repeats: int = 5
    seed: int = 42
    prompt_format: str = "chatml"
    prompt_file: str | None = None
    timeout_seconds: int = 600
    n_gpu_layers: tuple[int, ...] = (0,)
    n_batch: int | None = None
    n_ubatch: int | None = None
    type_k: str = "f16"
    type_v: str = "f16"
    flash_attn: bool = False
    validate_logits: bool = False

    def __post_init__(self) -> None:
        if not self.threads or min(self.threads) < 1 or len(set(self.threads)) != len(self.threads):
            raise ValueError("threads must be distinct positive integers")
        if self.context < 32 or not 2 <= self.max_tokens < self.context:
            raise ValueError("context must be >= 32; max-tokens must be >= 2 and smaller than context")
        if self.warmup < 1 or self.repeats < 2 or self.timeout_seconds < 1 or not 0 <= self.seed < 2**32:
            raise ValueError("warmup >= 1, repeats >= 2, positive timeout and a 32-bit seed are required")
        if self.prompt_format not in {"chatml", "raw", "auto"}:
            raise ValueError("prompt-format must be auto, chatml or raw")
        if not self.n_gpu_layers or len(set(self.n_gpu_layers)) != len(self.n_gpu_layers) or min(self.n_gpu_layers) < -1:
            raise ValueError("n-gpu-layers must be distinct integers >= -1")
        if len(self.n_gpu_layers) * len(self.threads) > 16:
            raise ValueError("GGUF search is capped at 16 configurations")
        if any(x is not None and x < 1 for x in (self.n_batch, self.n_ubatch)):
            raise ValueError("n-batch and n-ubatch must be positive")
        if self.n_ubatch and self.n_ubatch > (self.n_batch or self.context):
            raise ValueError("n-ubatch must not exceed n-batch")
        if self.type_k not in {"f16", "f32", "q8_0", "q4_0"} or self.type_v not in {"f16", "f32", "q8_0", "q4_0"}:
            raise ValueError("Unsupported KV-cache type")
        if self.type_v in {"q8_0", "q4_0"} and not self.flash_attn:
            raise ValueError("Quantized V cache requires explicit --flash-attn; backend support still must be tested")


def file_info(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".gguf":
        raise ValueError("model-file must be a GGUF file")
    with path.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF magic bytes")
        header = source.read(20)
        if len(header) != 20:
            raise ValueError("Truncated GGUF header")
        version, tensors, metadata = struct.unpack("<IQQ", header)
        if version not in {2, 3}:
            raise ValueError(f"Unsupported GGUF header version {version}")
        source.seek(0)
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"filename": path.name, "file_bytes": path.stat().st_size, "sha256": digest,
            "gguf_version": version, "tensor_count": tensors, "metadata_entry_count": metadata}


def prompts(config: GGUFConfig) -> list[dict[str, str]]:
    records = json.loads(Path(config.prompt_file).read_text(encoding="utf-8")) if config.prompt_file else list(DEFAULT_PROMPTS)
    if not isinstance(records, list) or not records:
        raise ValueError("Prompt file must contain a nonempty JSON list of {id, text} objects")
    result = []
    for item in records:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("id", "text")):
            raise ValueError("Each prompt requires nonempty string id and text")
        if config.prompt_format == "chatml" and any(token in item["text"] for token in ("<|im_start|>", "<|im_end|>")):
            raise ValueError("ChatML prompt text must not contain role-control tokens; use raw for preformatted prompts")
        rendered = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
                    if config.prompt_format == "chatml" else item["text"])
        record = {"id": item["id"], "text": item["text"], "rendered": rendered}
        if "expected_exact" in item:
            if not isinstance(item["expected_exact"], str):
                raise ValueError("expected_exact must be a string")
            record["expected_exact"] = item["expected_exact"]
        result.append(record)
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("Prompt ids must be unique")
    return result


class SampledRSS:
    """Sample this worker's process RSS; this is neither GPU nor model-only memory."""
    def __enter__(self) -> SampledRSS:
        import psutil

        self.process = psutil.Process()
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.stop = threading.Event()

        def sample() -> None:
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def measure_request(model: Any, tokens: list[int], maximum: int, seed: int) -> dict[str, Any]:
    # Explicitly discard the evaluated prefix so repeated prompts do not reuse it.
    model.reset()
    model.set_seed(seed)
    eos = model.token_eos()
    is_eog = lambda token: token == eos
    vocab = getattr(getattr(model, "_model", None), "vocab", None)
    if vocab is not None:
        import llama_cpp
        checker = getattr(llama_cpp, "llama_vocab_is_eog", None)
        if callable(checker):
            is_eog = lambda token: token == eos or bool(checker(vocab, token))
    generated: list[int] = []
    timestamps: list[int] = []
    finish = "length"
    generator = model.generate(tokens, reset=True, temp=0.0, top_k=0, top_p=1.0,
                               min_p=0.0, repeat_penalty=1.0)
    start = time.perf_counter_ns()
    end = start
    try:
        for token in generator:
            end = time.perf_counter_ns()
            if is_eog(token):
                finish = "eos" if token == eos else "eog"
                break
            generated.append(int(token))
            timestamps.append(end)
            if len(generated) >= maximum:
                break
    finally:
        generator.close()
    elapsed = (end - start) / 1e9
    if elapsed <= 0:
        raise ValueError("Nonpositive measured request duration")
    decode = (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) >= 2 else None
    return {"prompt_token_ids": tokens, "generated_token_ids": generated,
            "prompt_tokens": len(tokens), "generated_tokens": len(generated), "finish_reason": finish,
            "request_latency_ms": elapsed * 1000,
            "time_to_first_token_ms": (timestamps[0] - start) / 1e6 if timestamps else None,
            "decode_seconds": decode, "decode_intervals": max(len(generated) - 1, 0),
            "text": model.detokenize(generated).decode("utf-8", errors="replace")}


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("No timed samples")
    latencies = [item["request_latency_ms"] for item in samples]
    if not np.isfinite(latencies).all() or min(latencies) <= 0:
        raise ValueError("Invalid request latencies")
    first = [item["time_to_first_token_ms"] for item in samples if item["time_to_first_token_ms"] is not None]
    decode = [item for item in samples if item["decode_seconds"] is not None and item["decode_seconds"] > 0]
    intervals = sum(item["decode_intervals"] for item in decode)
    seconds = sum(item["decode_seconds"] for item in decode)
    total = sum(item["generated_tokens"] for item in samples)
    return {"request_latency_median_ms": statistics.median(latencies),
            "request_latency_p95_ms": float(np.percentile(latencies, 95)),
            "time_to_first_token_median_ms": statistics.median(first) if first else None,
            "time_to_first_token_p95_ms": float(np.percentile(first, 95)) if first else None,
            "decode_tokens_per_second": intervals / seconds if seconds else None,
            "overall_tokens_per_second": total / (sum(latencies) / 1000),
            "requests": len(samples), "generated_tokens": total}


def runtime_parameters(config: GGUFConfig, threads: int, layers: int, backend: Any) -> dict[str, Any]:
    signature = inspect.signature(backend.Llama.__init__).parameters
    parameters = {"n_ctx": config.context, "n_batch": config.n_batch or config.context,
                  "n_threads": threads, "n_threads_batch": threads, "n_gpu_layers": layers,
                  "offload_kqv": layers != 0, "seed": config.seed, "verbose": False}
    if layers and not (hasattr(backend, "llama_supports_gpu_offload") and backend.llama_supports_gpu_offload()):
        raise ValueError("SKIPPED — INCOMPATIBLE: installed llama.cpp has no GPU offload support")
    if config.n_ubatch is not None:
        parameters["n_ubatch"] = config.n_ubatch
    if config.flash_attn:
        parameters["flash_attn"] = True
    for argument, kind in (("type_k", config.type_k), ("type_v", config.type_v)):
        if argument in signature:
            constant = "GGML_TYPE_" + kind.upper()
            if not hasattr(backend, constant):
                raise ValueError(f"Backend lacks {constant}")
            parameters[argument] = getattr(backend, constant)
        elif kind != "f16":
            raise ValueError(f"Backend does not expose {argument}")
    missing = set(parameters) - set(signature)
    if missing:
        raise ValueError(f"Backend does not expose requested parameters: {sorted(missing)}")
    return parameters


def render_prompt(model: Any, record: dict[str, str], config: GGUFConfig) -> str:
    if config.prompt_format != "auto":
        return record["rendered"]
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    template = model.metadata.get("tokenizer.chat_template")
    if not template:
        raise ValueError("GGUF has no supported chat template; supply --prompt-format raw")
    bos = model.detokenize([model.token_bos()], special=True).decode("utf-8") if model.token_bos() >= 0 else ""
    eos = model.detokenize([model.token_eos()], special=True).decode("utf-8") if model.token_eos() >= 0 else ""
    formatter = Jinja2ChatFormatter(template=template, bos_token=bos, eos_token=eos)
    return formatter(messages=[{"role": "user", "content": record["text"]}]).prompt


def worker(config: GGUFConfig, threads: int, records: list[dict[str, str]], layers: int = 0,
           directory: Path | None = None) -> dict[str, Any]:
    import llama_cpp

    parameters = runtime_parameters(config, threads, layers, llama_cpp)
    with SampledRSS() as memory:
        start = time.perf_counter()
        model = llama_cpp.Llama(model_path=config.model_file, **parameters)
        effective = {"n_ctx": model.n_ctx(), "n_batch": model.n_batch,
                     "n_ubatch": getattr(getattr(model, "context_params", None), "n_ubatch", None)}
        setup = time.perf_counter() - start
        loaded_rss = memory.process.memory_info().rss
        try:
            metadata = model.metadata
            if config.prompt_format == "chatml":
                template = metadata.get("tokenizer.chat_template", "")
                if not all(marker in template for marker in ("<|im_start|>", "<|im_end|>")):
                    raise ValueError("This GGUF does not declare a ChatML template; supply raw formatted prompts")
            rendered = [{**item, "rendered": render_prompt(model, item, config)} for item in records]
            prepared = [(item, model.tokenize(item["rendered"].encode("utf-8"), add_bos=False, special=True)) for item in rendered]
            if any(not ids or len(ids) + config.max_tokens > config.context for _, ids in prepared):
                raise ValueError("Prompt plus max-tokens exceeds context (or prompt tokenizes to empty)")
            # A distinct prompt-only pass avoids mislabeling TTFT as prefill time.
            prefill = []
            logits = []
            for index, (item, ids) in enumerate(prepared):
                model.reset()
                start = time.perf_counter_ns()
                model.eval(ids)
                seconds = (time.perf_counter_ns() - start) / 1e9
                prefill.append({"prompt_id": item["id"], "tokens": len(ids), "seconds": seconds})
                if config.validate_logits:
                    getter = getattr(getattr(model, "_ctx", None), "get_logits", None)
                    if not callable(getter) or directory is None:
                        raise ValueError("Installed runtime does not expose prompt logits for validation")
                    values = np.ctypeslib.as_array(getter(), shape=(model.n_vocab(),)).copy()
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite prompt logits")
                    path = directory / f"prompt-logits-{index}.npy"
                    np.save(path, values, allow_pickle=False)
                    logits.append({"prompt_id": item["id"], "path": str(path.resolve())})
            for _ in range(config.warmup):
                for _, ids in prepared:
                    measure_request(model, ids, config.max_tokens, config.seed)
            samples = []
            with SampledRSS() as inference_memory:
                for repetition in range(config.repeats):
                    for item, ids in prepared:
                        result = measure_request(model, ids, config.max_tokens, config.seed)
                        if "expected_exact" in item:
                            result["task_quality_passed"] = result["text"].strip() == item["expected_exact"].strip()
                        samples.append({"prompt_id": item["id"], "repetition": repetition + 1, **result})
            info = {key: value for key, value in metadata.items()
                    if key.startswith(("general.", "llama.")) or key == "tokenizer.chat_template"}
        finally:
            model.close()
    quality = [s["task_quality_passed"] for s in samples if "task_quality_passed" in s]
    return {"threads": threads, "n_gpu_layers": layers, "status": "ok", "device": "cpu" if layers == 0 else "gpu-offload-requested",
            "backend": "llama-cpp-python", "runtime_parameters": parameters,
            "effective_runtime_parameters": effective,
            "gpu_offload_verification": "NOT VERIFIED" if layers else "not requested",
            "runtime_version": llama_cpp.__version__, "system_info": llama_cpp.llama_print_system_info().decode(),
            "model_metadata": info, "setup_seconds": setup, "samples": samples, "rendered_prompts": rendered,
            "prefill_samples": prefill, "prompt_logits": logits,
            "quality": {"evaluated_requests": len(quality), "passed_requests": sum(quality),
                        "exact_answer_rate": sum(quality) / len(quality) if quality else None,
                        "note": "Only predefined expected_exact tasks; token agreement alone is not quality."},
            "kv_cache": {"type_k": config.type_k, "type_v": config.type_v,
                         "requested_context": config.context, "context": effective["n_ctx"],
                         "exact_cache_bytes": None, "note": "Backend has no stable cache-allocation size API; RSS includes more than KV cache."},
            "metrics": {**aggregate(samples), "sampled_process_rss_peak_bytes": memory.peak,
                        "process_rss_before_load_bytes": memory.before, "process_rss_after_load_bytes": loaded_rss,
                        "model_loading_rss_delta_bytes": loaded_rss - memory.before,
                        "sampled_inference_rss_peak_bytes": inference_memory.peak,
                        "process_rss_after_close_bytes": memory.process.memory_info().rss,
                        "prompt_processing_tokens_per_second": sum(p["tokens"] for p in prefill) / sum(p["seconds"] for p in prefill),
                        "total_generation_seconds": sum(s["request_latency_ms"] for s in samples) / 1000,
                        "peak_gpu_memory_bytes": None, "sampled_process_vram_peak_bytes": None}}


def validate_runs(runs: list[dict[str, Any]]) -> None:
    """Validate greedy token sequences, not semantic accuracy or float logits."""
    baseline = runs[0]
    if baseline["status"] != "ok":
        return
    reference = {item["prompt_id"]: item for item in baseline["samples"] if item["repetition"] == 1}
    for run in runs:
        if run["status"] != "ok":
            continue
        differences = []
        for sample in run["samples"]:
            ref = reference[sample["prompt_id"]]
            if any(sample[key] != ref[key] for key in ("prompt_token_ids", "generated_token_ids", "finish_reason", "text")):
                differences.append({"prompt_id": sample["prompt_id"], "repetition": sample["repetition"]})
        passed = not differences
        logit_validation = {"status": "NOT VERIFIED", "reason": "Prompt logits were not requested"}
        if run.get("prompt_logits") and baseline.get("prompt_logits"):
            max_abs = max_rel = sum_abs = 0.0
            count = 0
            numerical_passed = True
            for ref_info, out_info in zip(baseline["prompt_logits"], run["prompt_logits"], strict=True):
                if ref_info["prompt_id"] != out_info["prompt_id"]:
                    raise ValueError("Prompt logits have different ids")
                ref = np.load(ref_info["path"], mmap_mode="r", allow_pickle=False)
                out = np.load(out_info["path"], mmap_mode="r", allow_pickle=False)
                try:
                    if ref.shape != out.shape:
                        raise ValueError("Prompt logits have different shapes")
                    for start in range(0, ref.size, 65536):
                        r, o = ref[start:start + 65536].astype(np.float64), out[start:start + 65536].astype(np.float64)
                        err = np.abs(r - o)
                        max_abs = max(max_abs, float(err.max()))
                        max_rel = max(max_rel, float((err / np.maximum(np.abs(r), 1e-8)).max()))
                        sum_abs += float(err.sum())
                        count += err.size
                        numerical_passed &= bool(np.isfinite(o).all() and np.all(err <= 1e-3 + 1e-2 * np.abs(r)))
                finally:
                    ref._mmap.close()
                    out._mmap.close()
            logit_validation = {"status": "PASS" if numerical_passed else "FAIL", "maximum_absolute_error": max_abs,
                                "mean_absolute_error": sum_abs / count, "maximum_relative_error": max_rel,
                                "atol": 1e-3, "rtol": 1e-2, "scope": "Last prompt-token logits against first runtime config, same quantized weights"}
            passed &= numerical_passed
        run["validation"] = {"passed": passed, "criterion": "Exact prompt/generated token ids, output text and finish reason",
                             "output_consistency": not differences,
                             "different_samples": differences, "accuracy_measured": False,
                             "prompt_logit_validation": logit_validation,
                             "note": "Agreement on these prompts is not semantic quality, accuracy or a logit-tolerance test."}
    if not baseline.get("validation", {}).get("passed"):
        return
    base = baseline["metrics"]
    for run in runs:
        if run.get("validation", {}).get("passed"):
            metrics = run["metrics"]
            run["comparison"] = {
                "comparable": True,
                "request_latency_speedup": base["request_latency_median_ms"] / metrics["request_latency_median_ms"],
                "decode_throughput_ratio": metrics["decode_tokens_per_second"] / base["decode_tokens_per_second"]
                if metrics["decode_tokens_per_second"] and base["decode_tokens_per_second"] else None}


def run_gguf(config: GGUFConfig, directory: Path) -> dict[str, Any]:
    for package in ("llama_cpp", "psutil"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError("GGUF dependencies missing. Install kernelixx[gguf]; see the README for prebuilt Windows wheels.")
    model = file_info(Path(config.model_file))
    records = prompts(config)
    runs = []
    for threads, layers in ((t, g) for t in config.threads for g in config.n_gpu_layers):
        task = directory / (f"threads-{threads}" if config.n_gpu_layers == (0,) else f"threads-{threads}-gpu-{layers}")
        task.mkdir(parents=True, exist_ok=False)
        payload = {"config": asdict(config), "threads": threads, "layers": layers, "prompts": records}
        source = task / "worker.json"
        source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run([sys.executable, "-m", "kernelixx.llm.gguf", "--worker", str(source.resolve())],
                                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=config.timeout_seconds)
            (task / "stdout.json").write_text(completed.stdout, encoding="utf-8")
            (task / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode:
                raise RuntimeError(f"GGUF worker exited {completed.returncode}; see {task / 'stderr.txt'}")
            result = json.loads(completed.stdout)
            if result["threads"] != threads or result.get("n_gpu_layers", 0) != layers:
                raise ValueError("Worker returned a different configuration")
            runs.append(result)
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as exc:
            runs.append({"threads": threads, "n_gpu_layers": layers, "status": "failed", "reason": str(exc)})
    if file_info(Path(config.model_file))["sha256"] != model["sha256"]:
        raise ValueError("Model file changed during the experiment")
    validate_runs(runs)
    return {"schema_version": 1, "experiment": "gguf", "model": model,
            "hardware": {"python": platform.python_version(), "platform": platform.platform(),
                         "processor": platform.processor(), "device": "cpu" if config.n_gpu_layers == (0,) else "mixed requested configurations"},
            "config": asdict(config), "prompts": records, "runs": runs,
            "limitations": [
                "Already-quantized GGUF: no FP32 reference or quantization savings are measured; optional prompt-logit checks compare runtime configs only.",
                "Native llama.cpp backend; PyTorch allocator measurements do not describe this model. GPU offload remains NOT VERIFIED unless native offload is established.",
                "Sampled process RSS at 10 ms is a worker working-set observation, not exact peak/model-only memory.",
                "Tokenization, model loading, output detokenization and warmup are outside request timing.",
                "TTFT is to the first non-EOG token; native end-of-turn/EOS markers are excluded. Decode uses N-1 intervals.",
                "Aggregate metrics mix the same fixed prompts with equal repetitions; short/EOS-only answers affect token counts.",
                "Each thread setting uses a fresh process, in listed order. One-session results are descriptive, not significance claims.",
                "Greedy token agreement on the supplied prompts is not real task accuracy or proof of general answer quality."]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated GGUF worker")
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.worker.read_text(encoding="utf-8"))
    payload["config"]["threads"] = tuple(payload["config"]["threads"])
    payload["config"]["n_gpu_layers"] = tuple(payload["config"].get("n_gpu_layers", [0]))
    try:
        result = worker(GGUFConfig(**payload["config"]), payload["threads"], payload["prompts"], payload.get("layers", 0), args.worker.parent)
    except ValueError as exc:
        if not str(exc).startswith("SKIPPED"):
            raise
        result = {"threads": payload["threads"], "n_gpu_layers": payload.get("layers", 0),
                  "status": "skipped", "status_detail": "SKIPPED — INCOMPATIBLE", "reason": str(exc)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

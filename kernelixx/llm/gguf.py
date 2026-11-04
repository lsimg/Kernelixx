from __future__ import annotations

import argparse
import hashlib
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
    {"id": "arithmetic", "text": "What is 7 plus 5? Answer with the number only."},
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

    def __post_init__(self) -> None:
        if not self.threads or min(self.threads) < 1 or len(set(self.threads)) != len(self.threads):
            raise ValueError("threads must be distinct positive integers")
        if self.context < 32 or not 2 <= self.max_tokens < self.context:
            raise ValueError("context must be >= 32; max-tokens must be >= 2 and smaller than context")
        if self.warmup < 1 or self.repeats < 2 or self.timeout_seconds < 1 or not 0 <= self.seed < 2**32:
            raise ValueError("warmup >= 1, repeats >= 2, positive timeout and a 32-bit seed are required")
        if self.prompt_format not in {"chatml", "raw"}:
            raise ValueError("prompt-format must be chatml or raw")


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
        result.append({"id": item["id"], "text": item["text"], "rendered": rendered})
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
            if token == model.token_eos():
                finish = "eos"
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


def worker(config: GGUFConfig, threads: int, records: list[dict[str, str]]) -> dict[str, Any]:
    import llama_cpp

    with SampledRSS() as memory:
        start = time.perf_counter()
        model = llama_cpp.Llama(model_path=config.model_file, n_ctx=config.context, n_batch=config.context,
                               n_threads=threads, n_threads_batch=threads, n_gpu_layers=0,
                               offload_kqv=False, seed=config.seed, verbose=False)
        setup = time.perf_counter() - start
        try:
            metadata = model.metadata
            if config.prompt_format == "chatml":
                template = metadata.get("tokenizer.chat_template", "")
                if not all(marker in template for marker in ("<|im_start|>", "<|im_end|>")):
                    raise ValueError("This GGUF does not declare a ChatML template; supply raw formatted prompts")
            # Raw prompts are fully formatted by the caller, including BOS if needed.
            prepared = [(item, model.tokenize(item["rendered"].encode("utf-8"),
                                            add_bos=False, special=True)) for item in records]
            if any(not ids or len(ids) + config.max_tokens > config.context for _, ids in prepared):
                raise ValueError("Prompt plus max-tokens exceeds context (or prompt tokenizes to empty)")
            for _ in range(config.warmup):
                for _, ids in prepared:
                    measure_request(model, ids, config.max_tokens, config.seed)
            samples = []
            for repetition in range(config.repeats):
                for item, ids in prepared:
                    samples.append({"prompt_id": item["id"], "repetition": repetition + 1,
                                    **measure_request(model, ids, config.max_tokens, config.seed)})
            info = {key: value for key, value in metadata.items()
                    if key.startswith(("general.", "llama.")) or key == "tokenizer.chat_template"}
        finally:
            model.close()
    return {"threads": threads, "status": "ok", "device": "cpu", "backend": "llama-cpp-python",
            "runtime_version": llama_cpp.__version__, "system_info": llama_cpp.llama_print_system_info().decode(),
            "model_metadata": info, "setup_seconds": setup, "samples": samples,
            "metrics": {**aggregate(samples), "sampled_process_rss_peak_bytes": memory.peak,
                        "process_rss_before_load_bytes": memory.before,
                        "peak_gpu_memory_bytes": None}}


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
            if any(sample[key] != ref[key] for key in ("prompt_token_ids", "generated_token_ids", "finish_reason")):
                differences.append({"prompt_id": sample["prompt_id"], "repetition": sample["repetition"]})
        passed = not differences
        run["validation"] = {"passed": passed, "criterion": "Exact prompt/generated token ids and finish reason",
                             "different_samples": differences, "accuracy_measured": False,
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
    for threads in config.threads:
        task = directory / f"threads-{threads}"
        task.mkdir(parents=True, exist_ok=False)
        payload = {"config": asdict(config), "threads": threads, "prompts": records}
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
            if result["threads"] != threads or result["device"] != "cpu":
                raise ValueError("Worker returned a different configuration")
            runs.append(result)
        except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as exc:
            runs.append({"threads": threads, "status": "failed", "reason": str(exc)})
    if file_info(Path(config.model_file))["sha256"] != model["sha256"]:
        raise ValueError("Model file changed during the experiment")
    validate_runs(runs)
    return {"schema_version": 1, "experiment": "gguf", "model": model,
            "hardware": {"python": platform.python_version(), "platform": platform.platform(),
                         "processor": platform.processor(), "device": "cpu"},
            "config": asdict(config), "prompts": records, "runs": runs,
            "limitations": [
                "Already-quantized GGUF: no FP32 reference, quantization savings or numerical logit validation are measured.",
                "CPU-only native backend. PyTorch/CUDA optimizations and GPU allocator measurements do not apply.",
                "Sampled process RSS at 10 ms is a worker working-set observation, not exact peak/model-only memory.",
                "Tokenization, model loading, output detokenization and warmup are outside request timing.",
                "TTFT is to the first non-EOS generated token. Decode rate uses intervals between non-EOS tokens (N-1).",
                "Aggregate metrics mix the same fixed prompts with equal repetitions; short/EOS-only answers affect token counts.",
                "Each thread setting uses a fresh process, in listed order. One-session results are descriptive, not significance claims.",
                "Greedy token agreement on the supplied prompts is not real task accuracy or proof of general answer quality."]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal isolated GGUF worker")
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.worker.read_text(encoding="utf-8"))
    payload["config"]["threads"] = tuple(payload["config"]["threads"])
    result = worker(GGUFConfig(**payload["config"]), payload["threads"], payload["prompts"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

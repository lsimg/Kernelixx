# Memory profiling

Install `pip install -e ".[memory]"`; optional `.[nvml]` adds nvidia-ml-py.
No GPU library is required for CPU execution.

```bash
python -m kernelixx profile-memory --model resnet50 --device cuda --input-shape 8,3,224,224
python -m kernelixx profile-memory --model examples.custom_model:load_model --device cpu --input-shape 8,16 --json
```

JSON is authoritative. `model_loading_memory` covers loading the original CPU
model. `baseline.memory` and each candidate's `metrics.memory` cover a separate
post-warmup forward; the sampler does not run during timed latency repetitions.
Run each CLI invocation in a fresh process when comparing CPU memory. The
`benchmark-repeat` command automates fresh-process comparison.

| Field | Meaning |
|---|---|
| process_rss_before_bytes / current_bytes | Process resident working set at scope boundaries |
| process_rss_delta_bytes | End minus beginning RSS; can be negative |
| sampled_process_rss_peak_bytes | Highest observed scope RSS at 10 ms sampling plus endpoints |
| os_process_lifetime_peak_rss_bytes | OS high-water mark, including earlier phases; not resettable inference peak |
| torch_allocated_* / peak_allocated_bytes | Live PyTorch CUDA allocations and scoped maximum |
| torch_reserved_* / peak_reserved_bytes | Memory managed by the CUDA allocator and scoped maximum |
| torch_transient_allocated_bytes | Scoped peak allocated minus allocated at scope start |
| process_vram_* / sampled_process_vram_peak_bytes | Optional NVML compute-process memory on the CUDA device mapped by UUID |

All byte fields are distinct. Missing sources yield null and notes, never zero.
Windows WDDM often cannot expose process VRAM; requested VRAM limits then cannot
be verified and the candidate cannot be accepted. CPU measurements do not become
CUDA measurements. `peak_gpu_memory_bytes` remains a backward-compatible alias
for PyTorch peak allocated bytes; it is not full process VRAM.
`cpu_peak_memory_bytes` remains null because an exact scoped CPU peak is unknown;
use `sampled_peak_ram_bytes` with its documented sampling limits.

RSS includes the interpreter, native libraries, mapped pages, source model and
runtime. A sampled phase peak is not model size or an exact instantaneous peak.
OS lifetime peaks include startup and earlier allocations. A low/zero RSS delta
does not prove absence of allocations or leaks. Samplers use finally cleanup and
join their threads even on exceptions. CUDA peaks use PyTorch high-water counters;
they are scoped to the current process/device allocator, not solely the runner.

Datasets can be NPZ (existing interface), or a directory containing float32
`inputs.npy` and int64 `labels.npy`. NPY files are mapped read-only; only the current
batch is copied. NPZ members must first be streamed to temporary NPY files because
NPZ is a ZIP container. This uses disk proportional to the uncompressed dataset.
Reference outputs are also spooled per batch to temporary disk. Temporary storage
must have enough space; errors remain visible and maps/files are closed on exit.
Periodic RSS observation may include mapped file pages accumulated by the OS.

Validation scratch work is limited by `--validation-chunk-elements` (default
65536). Maximum error is a maximum, mean error is weighted by element count, class
agreement/accuracy by example count. Accuracy drop subtracts integer counts first.
Changing chunk size can change the last bits of floating reduction sums, but not
the per-element tolerance policy. Large single-batch outputs still require memory
for that batch; lower the input batch size when necessary.

The legacy Python `make_batches` and `outputs` list APIs remain for existing
examples. They explicitly materialize their results; the optimizer uses the
streaming path. Tensor references are released before boundary cache cleanup.
`empty_cache()` is not used to conceal retained live tensors. No inference
recomputation was added; backward checkpointing needs a separate training study.

References: [PyTorch allocator semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management),
[NumPy loading and mmap](https://numpy.org/doc/stable/reference/generated/numpy.load.html),
[psutil memory fields](https://psutil.readthedocs.io/stable/#psutil.Process.memory_info).

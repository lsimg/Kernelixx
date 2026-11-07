# Benchmark methodology

## Timing scopes

Every run fixes seeds, weights, input shape, thread count, deterministic settings
and tolerance policy. CUDA wall-clock calls synchronize before/after each forward.
Transfer, preprocessing and validation are outside forward latency.

* Setup: model transformation/session construction; lazy Inductor compilation is
  not hidden here.
* Cold-start: first forward after runner construction, including lazy compilation.
  This is model-forward cold start, not Python process startup.
* Steady-state: median of individual post-warmup forward calls. p95 is an empirical
  percentile of these calls, not an independent-trial confidence interval.
* Throughput: batch size times repetitions divided by summed measured seconds.
  It counts examples, not batches. Changing batch size changes the workload.
* End-to-end runtime: the whole experiment lifecycle, including loading, data,
  transforms, validation, memory observations and runner cleanup; report writing
  and process startup are excluded unless explicitly measured by a harness.
* Model size: serialized bytes and weight bytes remain separate from RSS/VRAM.

An additional post-warmup forward measures memory separately from timed latency.
Reports preserve all raw timings and explicit scopes. Latency comparisons require
matching actual device, input shape, batch and timing policy.

The first/last thirds of at least ten timings are compared as a drift diagnostic.
An absolute median drift above 10% is UNSTABLE; fewer than ten timings give
INSUFFICIENT. `post_warmup_latency_median_ms` and raw medians remain available,
but `steady_state_latency_ms` is null unless STABLE_OBSERVED. Independent selection
returns INCONCLUSIVE when either side lacks observed stability. Increase warmup
and repetitions, then rerun; a higher warmup count alone is not proof of stability.
This is an engineering diagnostic, not a statistical stationarity/significance test.

## Independent paired trials

```bash
python -m kernelixx benchmark-repeat --model resnet18 --device cuda --input-shape 1,3,64,64 --pipeline fp16 --trials 3 --warmup 5 --repeats 30
python -m kernelixx optimize-pipeline --model resnet50 --device cpu --pipelines onnx onnx+quantization --trials 3 --max-candidates 2 --budget-seconds 900
```

A separate baseline worker creates reference outputs on disk. Each measured
baseline/candidate then runs in its own new Python process. Pair order alternates
baseline→candidate / candidate→baseline. Every worker validates the complete
stream against the fixed reference, checks source state/input/dataset hashes and
reports all observations. Independent pairs, not repeated forwards within a
worker, are the unit of repeatability. The reference worker is not counted as a
paired observation. Worker stdout/stderr and completed-run records are retained.

At least three complete pairs and a gain above `--min-improvement` in every pair
are needed for ACCEPT. All pairs below the margin give REJECT; mixed evidence,
insufficient pairs, timeouts or missing required resource measurements give
INCONCLUSIVE. Failed numerical/quality checks cannot be accepted. This is a
conservative engineering policy, not a test of statistical significance.
No confidence interval is fabricated from three observations.

Resource constraints use the OS lifetime process RSS peak at the inference
observation when available, falling back explicitly to sampled post-warmup RSS;
VRAM uses optional sampled NVML process memory. These are observed limits, not a
guarantee against an unobserved later/transient peak. Allocated bytes cannot
substitute for a full-process VRAM limit. If a limit
is exceeded the result is REJECT, even if latency improves. For memory/balanced,
the existing objective remains CUDA allocated-peak based and is unavailable on
CPU; process RSS is separately reported and can be constrained. A CPU memory
objective based on noisy scoped RSS is deliberately not silently substituted.

Search is capped at six explicit known chains, at most twenty paired trials per
candidate, a total elapsed budget, and individual worker timeouts. The in-process
preliminary transformation budget is checked between candidates; it cannot
preempt an individual native export/compile call. Repeated workers are bounded
by remaining budget. Do not run other benchmarks/tests concurrently with a study.

The old `select_best` helper computes descriptive single-session scores for API
compatibility. CLI single-session gains remain provisional and retain baseline;
`optimize-pipeline` performs independent confirmation. Baseline preservation is
an expected successful result, not evidence that conversion failed.

## Original-versus-new implementation

`tools/compare_implementations.py` accepts two trusted source checkouts and runs
alternating fresh processes with the same ResNet50 dataset, weights, batch, seed
and timing. An external 2 ms sampler measures each complete experiment using
identical instrumentation for both versions. This differs from the new engine's
separate inference memory scope. Raw output and limitations belong in
[PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md). Archived historical reports
retain their original meaning and are not current-version measurements.

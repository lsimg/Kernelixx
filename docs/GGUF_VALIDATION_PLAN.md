# GGUF validation milestones

The [SmolLM2 CPU experiment](GGUF_SMOLLM2.md) establishes a working native runner,
repeatable greedy outputs within its session, and a four-thread candidate with
lower observed latency. The following experiments turn that reference into
decisions about defaults, workload coverage and additional backends.

**Status:** this is a proposed experiment plan. Its entries do not represent
completed measurements or implemented new commands. The existing raw run and
its [derived analysis](experiments/smollm2-135m-q4-cpu/analysis.json) are the evidence
used to choose these priorities.

## Priority and decision map

| Priority | Check | Question to resolve | Decision enabled |
|---|---|---|---|
| 1 | Paired repeatability | Does the 1/4-thread gain persist across independent blocks and sessions? | Evidence for a recommended execution setting |
| 2 | Thread/workload sweep | Where do more threads help as prompt and answer lengths change? | Workload-specific thread recommendations |
| 2 | Labeled answer checks | Are answers correct, and do they follow the requested format? | A task-quality gate alongside token agreement |
| 3 | Memory phases and sampling | How do context size and instrumentation affect observed working set? | Defensible memory budgets and reporting granularity |
| 4 | Matched precision variants | What is the speed/storage/quality tradeoff for the same source weights? | Evidence-based choice of GGUF precision |
| 5 | Native integration CI | Can the installed CLI reliably load, generate and save on supported platforms? | Repeatable backend delivery checks |
| 6 | GPU execution track | Which workload benefits from verified native GPU offload? | An explicitly labeled CPU/GPU deployment comparison |

Priorities 2 and 3 may be prepared alongside repeatability. Precision and GPU
studies reuse the workload/quality protocol established by the earlier stages.

## 1. Paired repeatability before choosing a default

Start with five independent paired blocks, then repeat in a second session.
Run a fresh worker for each setting and counterbalance order: 1/4, 4/1, and so on.
Retain the current file hash, runtime, prompts, context, token limit, seed,
warmup and prompt/decode thread definitions. Record observable power/CPU state
and background-load conditions; retain all runs, including slower outcomes.

Calculate the latency ratio for each matched block and prompt, then report paired
ratios, their median and observed spread. Orient every ratio as latency at one
thread divided by latency at four threads, regardless of execution order.
Keep within-block request repetition
separate from the number of independent blocks. The current five repetitions
per prompt do not become additional independent sessions in this analysis.

**Proposed decision rule:** carry a setting forward when the median paired ratio
exceeds a declared practical margin (initial proposal: 1.01), the output policy
passes, and the full spread is published. A stronger interval-based claim would
also require a stated estimator and confidence level, with resampling at the
paired-block level. Do not infer significance from pooled requests or tokens.
Declare workload-specific tail-latency and memory budgets before recommending
a setting when those budgets are part of the deployment decision.

**Deliverable:** block/order metadata, per-prompt raw records, paired summaries
and a decision note linking the recommendation to those observations.

## 2. Workload length and thread-count mapping

Evaluate thread counts 1, 2, 4 and 8 where the available CPU/affinity permits.
Use the same source text for each comparison. Group measured prompt lengths
around 32, 128 and 256 tokens, then add longer context settings within the
model's declared capacity. Vary output caps such as 16, 64 and 128 tokens while
checking `prompt_tokens + max_tokens <= context`.

Actual EOS-stopped output lengths remain part of the record. Compare settings
within each workload cell before presenting a declared weighted composite.
Separate TTFT, decode throughput, full-request latency and sampled working set.
Increase observations per cell for a tail-latency study and publish the count
behind each p95; five repeats is a useful pilot rather than a tail-stability rule.

**Decision:** recommend thread settings by observed prompt/output regime rather
than assuming four threads is globally optimal. Short replies, long continuations
and larger contexts may favor different resource allocations.

**Deliverable:** a thread-by-workload table, measured token lengths, per-cell
sample counts and the same token-agreement policy used in the reference run.

## 3. Task quality and instruction compliance

Build a fixed public prompt set with separate content and format labels. Include
arithmetic, short factual answers, constrained lists and summarization. Start
with English, matching the model's documented focus; evaluate Russian prompts
as a separately reported user-language subset with its own references.

Define grading rules before inspecting new outputs: number-only answers can
have separate arithmetic and numeric-format checks; comma-separated lists can
have separate accepted-content and delimiter/length checks. Record exact-match
or normalization rules, accepted alternatives and any human-reviewed rubric.

Continue reporting exact token agreement for same-file execution-setting
comparisons. Add task scores as a different layer. If generated sequences differ,
retain the differences and assess them under the predeclared task policy rather
than changing the agreement criterion after seeing a favorable speed result.

**Decision:** distinguish faster reproduction of the same answers from improved
task performance. The observed arithmetic/list responses already give concrete
cases for the first format-compliance checks.

**Deliverable:** versioned prompts/labels, grading policy, per-example outcomes
and separate content/format aggregates with sample counts.

## 4. Memory phases and observer overhead

Measure before load, after load, after prompt processing and during continuation.
Repeat the same workload at different context sizes. Keep the GGUF file size,
process working set and any available native model/KV-buffer accounting in
separate fields, with explicit units and measurement methods.

Compare the current 10 ms sampler with other intervals and with sampling disabled
under otherwise identical execution. This estimates instrumentation overhead and
the sensitivity of observed peaks to sampling frequency. Native buffer sizes or
platform-specific allocation counters can complement RSS when their semantics
are verified; none should silently replace another metric.

**Decision:** choose an appropriate sampling mode and resource budget for a given
context/workload, supported by phase measurements and observer-cost evidence.

**Deliverable:** phase/context memory tables, sampler settings and overhead
comparison, with sampled working-set peaks labeled as observations.

## 5. Matched-source precision comparison

Establish one source-weight revision and tokenizer/template, and record the
converter revision and commands used to produce matched F16, Q8 and Q4 variants
supported by the selected runtime. Hash every artifact. Keep native backend,
device, workload, context and reporting instruments fixed.

Compare storage, sampled working set, TTFT, decode/request time and the labeled
task outcomes. Choose task-specific acceptance thresholds and an adequate labeled
sample size before testing. Exact token identity can be reported descriptively,
but quantization quality requires its own task policy; optionally add held-out
text/perplexity or logit comparisons when the backend exposes the required data.

**Decision:** choose a precision based on a measured resource/quality tradeoff
attributed to matched artifacts. The current Q4_K_M run supplies one reference
point; it does not serve as evidence for an unmeasured F16/Q8 comparison.

**Deliverable:** conversion lineage, hashes, settings and a joint resource/quality
table, with every accepted or rejected variant retained.

## 6. Installed native-backend integration

Add an opt-in integration lane with a small, pinned, appropriately licensed GGUF
fixture. Check loading, bounded generation, saved-report reading and clean error
handling from an installed wheel on the supported Windows/Linux Python matrix.
Keep private user models outside CI and record the fixture/runtime hashes.

Use functional contracts for pass/fail: successful generation, valid finite
metrics, correct report shape, controlled worker failure/timeout and readable
artifacts. Performance remains an experiment result rather than a fixed CI-speed
threshold. Report platform-specific generated outputs; distinguish cross-platform
observations from agreement within a matched execution environment.

**Decision:** qualify native runtime delivery independently of model speed or
quality. The existing 90-test matrix remains the base, complemented by local
real-model evidence and the proposed optional fixture lane.

## 7. Native GPU track

Verify the native runtime's Windows/CUDA compatibility and actual offload before
adding GPU measurements. Record driver/runtime versions, layer placement,
offloaded-layer count, context, warmup, token policy and timing completion boundary.
Use a GPU memory instrument with documented scope; PyTorch allocator peaks do
not measure allocations owned by the native GGUF backend.

Present CPU and GPU as separate deployment configurations, with device and
offload settings visible. Any cross-device ratio must explicitly identify both
configurations. Apply the same task/sequence policy and retain a CPU reference.

**Decision:** identify workloads where GPU deployment is useful and select a
documented offload setting. CPU thread ratios are not reused as GPU evidence.

## Evidence and reporting contract

Each new study records its question, predeclared protocol, model/runtime identity,
raw measurements, output-validation policy and decision. Publish completed
observations separately from this plan. Preserve earlier measurement records so
that later findings can be compared without rewriting the reference experiment.

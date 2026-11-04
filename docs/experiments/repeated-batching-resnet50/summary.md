# Repeated FP32 batching study

| Trial | Batch 1 images/s | Batch 8 images/s | Ratio |
|---:|---:|---:|---:|
| 1 | 70.0747 | 264.8099 | 3.778964 |
| 2 | 69.1197 | 321.1945 | 4.646931 |
| 3 | 66.8117 | 326.1665 | 4.881876 |

```json
{
  "trial_count": 3,
  "batch1_images_per_second": {
    "median": 69.11969706772126,
    "minimum": 66.81171043590899,
    "maximum": 70.0747276896082,
    "sample_standard_deviation": 1.6776058028881806
  },
  "batch8_images_per_second": {
    "median": 321.1944837416973,
    "minimum": 264.80987280056365,
    "maximum": 326.1665039231715,
    "sample_standard_deviation": 34.07976310164084
  },
  "paired_throughput_ratios": [
    3.778963993602988,
    4.6469312998724694,
    4.881876272813819
  ],
  "paired_ratio_statistics": {
    "median": 4.6469312998724694,
    "minimum": 3.778963993602988,
    "maximum": 4.881876272813819,
    "sample_standard_deviation": 0.5809449028231422
  },
  "trials_with_batch8_higher_throughput": 3,
  "note": "Descriptive statistics of whole-process trials; no confidence interval or significance claim."
}
```

Throughput ratios compare different batch sizes; they are not single-image latency speedups.
Fresh process per configuration; alternating order; all measured calls use the first fixed batch.
Queueing, file decoding, preprocessing, input transfers and process startup are excluded.
Accuracy covers only the supplied labeled subset. Three trials do not establish statistical significance.

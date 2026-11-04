# ResNet18 versus ResNet50: per-photo answers

Executed on 2026-11-01 (Asia/Qyzylorda), Windows 11, Python 3.12.10,
PyTorch 2.9.1+cu128. Pretrained FP32 inference used CPU, one thread, batch 1,
seed 42. No training or new timing comparison was performed.

## Result on the same 200 photos

| Configuration | Correct | Accuracy |
|---|---:|---:|
| ResNet18 DEFAULT / ImageNet1K V1 | 131/200 | 65.50% |
| ResNet50 DEFAULT / ImageNet1K V2 | 155/200 | 77.50% |

| Outcome | Photos |
|---|---:|
| ResNet50 corrected a ResNet18 mistake | 30 |
| ResNet18 was correct but ResNet50 was wrong | 6 |
| Both correct | 125 |
| Both wrong | 39 |

The net gain is **24 correct answers**, or **12 percentage points**. It does not
mean that only 24 mistakes were corrected: 30 fixes and six regressions occurred.

## Actual examples

Indices match the original ResNet18 review.

| Photo | Supplied label | ResNet18 | ResNet50 | Outcome |
|---|---|---|---|---|
| 10 | English springer | Bernese mountain dog | English springer | fixed |
| 19 | garbage truck | moving van | garbage truck | fixed |
| 31 | cassette player | spotlight | cassette player | fixed |
| 47 | chain saw | dogsled | chain saw | fixed |
| 1 | gas pump | gas pump | tobacco shop | regressed |
| 199 | English springer | English springer | Border collie | regressed |
| 21 | gas pump | vending machine | vending machine | both wrong |

"Wrong" means disagreement with the supplied label. Some images have ambiguous
or closely related categories. This is a small, class-balanced ten-category
subset, with argmax over all 1000 ImageNet outputs. Different pretrained weight
recipes and official resize settings are used; these results compare practical
configurations and do not isolate architecture depth or establish production accuracy.

## Reproduction

The ResNet18 review is described in [PREDICTION_REVIEW.md](PREDICTION_REVIEW.md).
After preparing both datasets as described in the README:

```powershell
$env:TORCH_HOME = Join-Path (Get-Location) 'results/torch-cache'
.\.venv\Scripts\python.exe -m examples.inspect_predictions --dataset results/datasets/imagenet-val-resnet50-200.npz --output results/prediction-review-resnet50
.\.venv\Scripts\python.exe -m examples.compare_predictions --resnet18-review results/prediction-review-resnet18/predictions.json --resnet50-review results/prediction-review-resnet50/predictions.json --output results/model-comparison-resnet18-resnet50
```

Choose new output paths if these directories exist. The reader verifies review
model/weights, dataset SHA-256, preprocessing, manifest order, labels, original photo
hashes, class names and individual correctness flags. Comparison joins by manifest
path plus SHA-256, so shuffled row order cannot silently compare different photos.

## Artifacts and checks

Local `results/model-comparison-resnet18-resnet50/` contains `comparison.json`,
`comparison.csv`, separate CSVs for all four outcomes, and `report.md`. Three PNGs
show all 30 fixes, one shows all six regressions, and four show the 39 shared errors.
Cards show the original photograph and both answers, with green for correct and
red for wrong. The separate model reviews show their exact preprocessed crops.
No GUI or server is created. Both fixed and regressed layouts were visually checked.

The full automated suite has **71 passing tests**. Six new checks verify all four
outcomes, reordered inputs, changed hashes/labels, duplicate/missing images, and
report generation including fixes and regressions when the net gain is zero.
The existing end-to-end review check also verifies the comparison reader's source
validation and rejection of inconsistent saved answers.
Full test output: [test-results-model-comparison.txt](test-results-model-comparison.txt).
All per-photo results are also archived as [JSON](experiments/model-comparison/comparison.json)
and [CSV](experiments/model-comparison/comparison.csv), without copying raw photographs.
Source archives are rebuilt after this follow-up; earlier release-install checks
remain historical evidence for their documented revision.

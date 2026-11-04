# User-selected images: real model answers

Executed on 2026-11-02 (Asia/Qyzylorda), Windows 11, Python 3.12.10,
PyTorch 2.9.1+cu128. Seven explicit, user-supplied PNG screenshots of apples were
read from their existing locations. FP32 inference used CPU, one thread, batch 1,
seed 42. The original images were not changed or copied into the source archives.

## Actual top answers

Numbers preserve the attachment order.

| Image | ResNet18 DEFAULT | ResNet50 DEFAULT |
|---|---|---|
| 1 | Granny Smith | Granny Smith |
| 2 | fig | Granny Smith |
| 3 | bell pepper | Granny Smith |
| 4 | Granny Smith | Granny Smith |
| 5 | chambered nautilus | Granny Smith |
| 6 | fig | banana |
| 7 | pomegranate | orange |

The `Granny Smith` class was selected on two images by ResNet18 and five by
ResNet50. This is a descriptive count of chosen classes, not an accuracy estimate.
The ImageNet-1K category metadata contains `Granny Smith` (index 948), but no
general `apple` class. These red-apple screenshots do not establish a correct
cultivar prediction. `pineapple` and `custard apple` are separate categories.
Exact ImageNet class labels were not supplied; `accuracy_measured` remains false.

ResNet18 and ResNet50 use different pretrained weight recipes and official resize
settings. These seven closely related screenshots are a small demonstration,
not a representative task evaluation or evidence that architecture depth alone
caused an improvement. No training, optimization or timing experiment occurred.

## Reproduction and output

```powershell
.\.venv\Scripts\python.exe -m examples.predict_images --images "C:\Pictures\apple.png" "C:\Pictures\another photo.jpg"
```

Replace the example paths with existing files. Explicit paths avoid scanning an
unrelated private image folder. The default output path is a new timestamped
directory under `results/user-images/`. This executed run used
`results/user-apples-20261102/`: `predictions.json`, `predictions.csv`, `report.md`
and `comparison-01.png`. JSON records input hashes/dimensions, preprocessing,
pretrained weights and five ranked classes with softmax scores per model/image.
The CSV was generated from those same measured outputs. Softmax scores are not
calibrated probabilities or measured accuracy. No fabricated labels or results
were inserted. Contact-sheet layout was inspected visually.

## Automated checks

The full suite has **74 passing tests**. Three added tests check unconstrained
class ranking/nonfinite output rejection; Unicode paths, image modes, ordering
and duplicate-path rejection; and actual preprocessing/report generation with
deterministic probe models. Cached checkpoint selection, cache-state restoration
after failures and overwrite protection are covered. Probe models in tests do not
replace the real pretrained inference reported above.
Full output: [test-results-user-images.txt](test-results-user-images.txt).

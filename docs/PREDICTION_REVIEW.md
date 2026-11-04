# Photo-by-photo prediction review

Executed locally on 2026-11-01 (Asia/Qyzylorda), Windows 11, Python 3.12.10,
PyTorch 2.9.1+cu128. Inference used CPU FP32, one thread, batch size 1, seed 42.
This follow-up makes the previously aggregated accuracy inspectable per photograph.

## Executed command

```powershell
.\.venv\Scripts\python.exe -m examples.inspect_predictions --dataset results/datasets/imagenet-val-resnet18-200.npz --onnx results/manual-demo/20261101T165242Z-62313a31/onnx/model.onnx --output results/prediction-review-resnet18
```

The ONNX path identifies the existing user experiment's exported model. To reproduce
elsewhere, prepare the dataset and run the example without `--onnx`, or supply an
export from a new CPU ONNX experiment. No external upload or model training occurs.

## Measured predictions

- ResNet18 DEFAULT pretrained weights: **131/200 correct (65.50%)**, **69 mistakes**.
- Saved ONNX model: **131/200 correct**, **zero class predictions changed**.
- ONNX numerical validation passes the existing default tolerances.
- All 200 photos are evaluated in their manifest order; argmax covers all 1000 classes.

Examples below are actual predictions, with indices matching `predictions.csv`:

| Photo index | Supplied label | PyTorch and ONNX prediction |
|---|---|---|
| 9 | church | monastery |
| 10 | English springer | Bernese mountain dog |
| 21 | gas pump | vending machine |
| 25 | English springer | Welsh springer spaniel |

"Mistake" means disagreement with the supplied class label, not a manual audit of
that annotation. Similar or ambiguous categories can disagree under this exact-label
metric. Results cover this ten-class 200-image subset only.

## Local artifacts

`results/prediction-review-resnet18/` contains every prediction in JSON and CSV,
the 69 mistakes in `errors.csv`, a Markdown report with source photo links, six
PNG sheets showing all mistakes, and a sheet of six correct examples. Each card
shows the original photo and the exact resized/cropped input, reconstructed from
the normalized NPZ tensor. PNGs and raw photos remain local under gitignored results.

The example verifies NPZ SHA-256, per-image SHA-256, manifest label order,
pretrained weights and preprocessing, and rejects missing/modified files or reused
output directories. It reports saved-model hashes when ONNX is supplied.

## Automated checks

The full suite has **65 passing tests**. Three added tests cover order and full-class
prediction mapping, label/hash/path mismatch detection, and an end-to-end review
with output files, contact sheets, ONNX agreement and overwrite protection. Test
models are deterministic local probes; the real pretrained run above is separate.
The PNG layout was inspected visually after generation. Earlier installation
checks in `RELEASE_VALIDATION.md` describe the previous 62-test revision; the
source archives are rebuilt after this follow-up.

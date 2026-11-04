# Contributing to Kernelixx

Use Python 3.11 or 3.12 and the PyTorch 2.9.x / torchvision 0.24.x pair documented
in the README. Install the desired CPU/CUDA wheels first, then:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[onnx,dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use local deterministic probes and random weights; they do not download
pretrained checkpoints. CUDA-only tests skip when CUDA is unavailable. To work
around a restricted temporary directory in a managed sandbox, use a new
`--basetemp results/pytest-NEW-ID` path. Do not reuse a path containing important
files, because pytest owns its temporary directory.

## Engineering expectations

- Keep the product CLI-only and responsibilities separated by module.
- Include real measurements for performance claims. Preserve negative outcomes,
  raw timings, seeds, versions, workload shape and validation settings.
- Compare identical devices/workloads; document changed batch sizes separately.
- Distinguish weight storage, allocator peaks, numerical similarity and task accuracy.
- Do not claim kernel fusion without compiler/profiler evidence.
- Add focused tests for correctness or lifecycle changes. Do not add fabricated
  benchmark snapshots or tests that merely restate implementation details.
- Update the README, architecture/history and changelog when changing scope.
- Keep credentials, environments, raw images/datasets and model binaries out of Git.

## Packaging

```powershell
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe tools/package_source.py
```

The source tool audits artifact inventories and compares packaged source bytes
with the checkout. Use `--overwrite` when intentionally rebuilding the generated
source ZIP of the same version. Install the wheel in an isolated environment and
run `tools/smoke_installed.py` with that environment's Python from outside the
checkout. CI repeats this on Windows/Linux with Python 3.11/3.12.

Public experiment copies may normalize paths and project branding, but measured
numbers must not be changed. Retain original local artifacts and document any
normalization. Private user-image originals are not publication assets.

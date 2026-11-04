# Kernelixx 0.1.0: package and installation validation

Validated locally on 2026-11-01 (Asia/Qyzylorda), Windows 11, 64-bit Python 3.12.10.
No external repository or package registry publication was performed during these checks.

## Build metadata and source contents

- Runtime support is bounded to PyTorch 2.9.x / torchvision 0.24.x, tested at 2.9.1 / 0.24.1.
- Setuptools >=77.0.3 is required for the SPDX license expression and license-file metadata.
- `build` is a development dependency; the runtime package remains a CLI.
- `MANIFEST.in` explicitly includes examples, tests, tools, documentation, dependency snapshots and CI.
- Wheel and source distribution build successfully; an offline wheel rebuild from the source distribution also passed.
- `tools/package_source.py` creates a separate source ZIP, audits all three inventories, and compares packaged source bytes with the current files to reject stale builds.
- Generated model/data directories and virtual environments are excluded. Archived metric reports remain in `docs/experiments/`.
- Artifact sizes, complete file lists and SHA-256 values are written to `dist/release-manifest.json` and `dist/SHA256SUMS.txt` for each build.

The wheel installs `kernelixx/` and its console command. Examples, tests, tools and experiment reports are available in the source archives/repository.

## Actual isolated installation

Created a new virtual environment under `results/release-check/clean-env` with
`include-system-site-packages = false`. Because the managed sandbox restricts its
default temporary directory, pip was bootstrapped explicitly and temporary build
files were directed into the workspace. This preparation does not change normal
installation instructions on an unrestricted Windows machine.

Installed the official CPU PyTorch pair, then the locally built wheel with
`onnx` and `dev` extras. The fresh resolver selected, among other dependencies:

| Dependency | Installed version |
|---|---|
| Python | 3.12.10 |
| torch | 2.9.1+cpu |
| torchvision | 0.24.1+cpu |
| numpy | 2.5.4 |
| rich | 15.0.0 |
| onnx | 1.23.2 |
| onnxruntime | 1.31.1 |
| pillow | 13.0.0 |
| pytest | 9.2.0 |
| setuptools in clean environment | 84.2.1 |
| build | 1.6.1 |

`pip check` reported **No broken requirements found**. The dependency snapshots
used for earlier performance experiments are retained unchanged; these newer
fresh-install versions are a separate compatibility check.

## Installed-wheel checks outside the checkout

`tools/smoke_installed.py` uses isolated Python mode, removes `PYTHONPATH` from
child environments, and runs from a separate working directory. It verifies
that both Kernelixx and PyTorch load from the tested Python prefix, and that
the package version matches installed metadata.

Nine real commands passed:

1. Installed module path/version probe.
2. `python -m kernelixx --help`.
3. Installed `kernelixx --version` console entry point.
4. `system-info --json`.
5. CPU ResNet18 analysis with real Profiler events.
6. ONNX export, inference, validation and JSON/Markdown/CSV output.
7. Saved-report `compare` round trip.
8. CPU FP16 request: explicit skip, report retained, expected exit code 2.
9. CPU small-CNN training: real FP32 iterations and explicit AMP skip.

These are functional checks with random weights and small synthetic inputs;
their timings are not used as performance evidence.

## Source-distribution tests

Unpacked the built source archive into a separate directory and ran pytest using
only the new environment's dependencies:

**59 passed, 3 skipped, 6 upstream deprecation warnings.**

The three skips require CUDA, which is deliberately absent from this CPU wheel.
They were exercised in the existing CUDA development environment, whose full
suite has **62 passing tests**.

[Full isolated CPU test output](test-results-clean-install.txt).

Rebuilding a wheel from the source archive succeeded with:

```powershell
python -m pip wheel --no-cache-dir --no-index --disable-pip-version-check --no-deps --no-build-isolation dist/kernelixx-0.1.0.tar.gz --wheel-dir results/release-check/rebuilt-offline
```

Build dependencies were already installed for this offline check. An earlier
attempt using the sandbox's default pip cache did not finish and was interrupted;
the workspace-temporary, no-cache offline run completed successfully. Its cause
was not established.

## Reproduce the installed-wheel check

After creating an isolated environment, installing the CPU or CUDA PyTorch pair,
and installing the built wheel with `[onnx,dev]`, run the checker with that
environment's Python:

```powershell
python -I tools/smoke_installed.py --output results/wheel-smoke-new
```

Choose a new output directory for each check. The checker executes child commands
with isolated imports and an output directory as their working directory, and
rejects an editable/source import outside the tested Python prefix.

## Remaining platform checks

- Local clean installation was tested on Python 3.12, Windows 11, CPU.
- Python 3.11 and Linux are configured in GitHub Actions but have not been run remotely here.
- CI builds distributions, audits them, installs the wheel and checks it outside the checkout.
- GPU performance/quality evidence comes from the existing CUDA environment and is documented separately.
- No PyPI availability, external GitHub publication, or cross-platform CI success is claimed.

## References

- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
- [License metadata and backend requirements](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
- [Build frontend: source archive then wheel](https://build.pypa.io/en/latest/how-to/basic-usage.html).

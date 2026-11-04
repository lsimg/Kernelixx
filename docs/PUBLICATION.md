# Kernelixx naming and publication checks

Prepared on 2026-11-04 (Asia/Qyzylorda), Windows 11, Python 3.12.10.
The public repository is `lsimg/Kernelixx`. This first public import contains
the implemented project and its documented local development history.

## Naming contract

| Item | Current name |
|---|---|
| Repository / application | Kernelixx |
| Python distribution / import | `kernelixx` |
| Module invocation | `python -m kernelixx` |
| Installed terminal command | `kernelixx` |
| Wheel | `kernelixx-0.1.0-py3-none-any.whl` |
| Source distribution | `kernelixx-0.1.0.tar.gz` |
| Source ZIP | `Kernelixx-0.1.0-source.zip` |

The package directory, imports, CLI parser, profiler marker, report branding,
install tools, examples, test imports and package metadata use the new name.
The previous command is replaced, so reinstall the package and use `kernelixx`.
Report schema version 1 remains readable; old local experiment results are
preserved and may be read with the new `compare` command.

## Source validation

The renamed project passes **74 tests** in the CUDA-enabled development
environment, including actual CUDA inference and AMP integration. Six upstream
warnings concern the deprecated quantization/legacy ONNX interfaces documented
in the README. The installed `kernelixx --version` entry point returns `0.1.0`.
Full output: [test-results-publication.txt](test-results-publication.txt).

The same source suite in the isolated CPU environment passes **71 tests**, with
**three CUDA-only skips** and six upstream warnings. Full output:
[test-results-publication-cpu.txt](test-results-publication-cpu.txt).

The new wheel was installed offline into the existing isolated CPU environment
(`include-system-site-packages=false`), replacing the former application package.
This reused the previously resolved CPU dependencies; it was not a new dependency
resolution. From outside the checkout, **all nine installed-wheel checks passed**:
installed import/metadata, module help, console version, system info, ResNet18
Profiler analysis, ONNX optimization/validation, saved-report comparison, explicit
CPU FP16 skip and real CPU training with AMP skip. The checker verified that
Kernelixx and PyTorch imported from that environment's prefix.
[Normalized smoke-check record](experiments/kernelixx-installation/smoke-result.json).
Functional checks use random weights and tiny synthetic inputs and are not new
performance evidence. The unchanged schema also reads pre-rename local reports.

CI is configured for Windows/Linux, Python 3.11/3.12. Current remote results belong
to the repository's Actions page; a local pass is not a remote CI result.

## Public content boundaries and evidence provenance

Source publication uses the packaging tool's explicit file allowlist plus the
empty `results/.gitkeep`. It includes code, examples, tests, CI, README files,
architecture, development history, changelog, contribution instructions, license
and archived numerical evidence. It excludes environments, caches, raw working
results, private images, datasets, checkpoints, ONNX binaries and credentials.

The original local experiment directories remain intact. Copies under `docs/`
normalize the old project name, machine-specific absolute paths and trailing
whitespace in text logs/CSV. Numeric
JSON values were checked against the prior local source archive: **all 26
previously archived JSON files retain their numbers**. Hashes that describe
original source data/review files continue to describe those original files;
the redacted public evidence copies are not claimed to have the same file hash.

No photographs from the user-image experiment are publication assets. The
public document records only indexed model answers and methodology. Downloadable
datasets and weights must be obtained separately using their upstream terms.

## Documentation map

- [README](../README.md): setup, commands, methods, methodology and results.
- [Russian guide](../README.ru.md): plain-language manual usage.
- [Architecture](ARCHITECTURE.md): modules, contracts, lifecycle and metrics.
- [Development history](DEVELOPMENT_HISTORY.md): phases, corrections, setbacks and evidence.
- [Changelog](../CHANGELOG.md): delivered features and implementation improvements.
- [Validation index](VALIDATION.md): all measured local experiment follow-ups.
- [Contributing](../CONTRIBUTING.md): checks, packaging and measurement standards.

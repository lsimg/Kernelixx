"""Exercise an installed wheel from outside the checkout (no model downloads)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve() if args.output else Path.cwd() / ("wheel-smoke-" + uuid.uuid4().hex[:8])
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    environment.pop("PYTHONPATH", None)
    checks: list[dict[str, Any]] = []

    def run(label: str, command: list[str], expected: int = 0, as_json: bool = False) -> Any:
        result = subprocess.run(command, cwd=output, env=environment, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        (output / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        checks.append({"name": label, "command": command, "exit_code": result.returncode,
                       "expected_exit_code": expected})
        if result.returncode != expected:
            raise RuntimeError(f"{label}: exit {result.returncode}, expected {expected}; see {output} logs")
        return json.loads(result.stdout) if as_json else result.stdout.strip()

    def cli(label: str, command: str, arguments: list[str], expected: int = 0) -> dict[str, Any]:
        return run(label, [sys.executable, "-I", "-m", "kernelixx", command, *arguments, "--json"], expected, True)

    try:
        output.mkdir(parents=True, exist_ok=False)
        probe = run("installed-path", [sys.executable, "-I", "-c",
            "import json,sys,importlib.metadata as m,kernelixx,torch; "
            "print(json.dumps({'module':kernelixx.__file__,'version':kernelixx.__version__,"
            "'metadata_version':m.version('kernelixx'),'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
            "'torch_file':torch.__file__,'torch_version':torch.__version__}))"], as_json=True)
        prefix = Path(probe["prefix"]).resolve()
        if not Path(probe["module"]).resolve().is_relative_to(prefix):
            raise ValueError("Kernelixx was loaded from the checkout/editable install, not the installed wheel")
        if not Path(probe["torch_file"]).resolve().is_relative_to(prefix):
            raise ValueError("PyTorch was loaded from outside the tested Python prefix")
        if probe["version"] != probe["metadata_version"]:
            raise ValueError("Package version and distribution metadata differ")
        run("module-help", [sys.executable, "-I", "-m", "kernelixx", "--help"])
        executable = Path(sysconfig.get_path("scripts")) / ("kernelixx.exe" if os.name == "nt" else "kernelixx")
        if run("console-version", [str(executable), "--version"]) != probe["version"]:
            raise ValueError("Console entry point returned an unexpected version")
        hardware = cli("system-info", "system-info", [])
        shared = ["--model", "resnet18", "--device", "cpu", "--input-shape", "1,3,32,32",
                  "--warmup", "1", "--repeats", "2", "--validation-batches", "1"]
        analyzed = cli("analyze", "analyze", [*shared, "--output", str(output / "analyze")])
        if not analyzed.get("analysis", {}).get("bottlenecks"):
            raise ValueError("Installed analyzer did not produce profiler events")
        optimized = cli("onnx", "auto-optimize", [*shared, "--methods", "onnx", "--output", str(output / "onnx")])
        candidate = optimized["candidates"][0]
        if candidate["status"] != "ok" or not candidate["validation"]["passed"]:
            raise ValueError("Installed ONNX path failed validation")
        for name in ("report.json", "report.md", "metrics.csv"):
            if not (Path(optimized["run_directory"]) / name).is_file():
                raise ValueError(f"Missing installed-run artifact: {name}")
        compared = cli("compare", "compare", ["--run-dir", str(output / "onnx/latest")])
        if compared["baseline"] != optimized["baseline"]:
            raise ValueError("Saved comparison changed baseline measurements")
        rejected = cli("cpu-fp16-skip", "optimize", [*shared, "--method", "fp16",
                       "--output", str(output / "unsupported")], expected=2)
        if rejected["candidates"][0]["status"] != "skipped":
            raise ValueError("CPU FP16 must be explicitly skipped")
        training = cli("training", "train-benchmark", ["--device", "cpu", "--steps", "2",
                       "--warmup", "1", "--batch-size", "2", "--output", str(output / "training")])
        if [run["status"] for run in training["runs"]] != ["ok", "skipped"]:
            raise ValueError("Expected real FP32 CPU training and an explicit AMP skip")
        result = {"passed": True, "probe": probe, "hardware": hardware, "checks": checks,
                  "note": "Functional smoke checks use random weights and small synthetic inputs; timings are not performance evidence."}
        (output / "smoke-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Installed wheel smoke checks passed ({len(checks)} commands).")
        print(f"Imported package: {probe['module']}")
        print(f"Result: {output / 'smoke-result.json'}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if output.is_dir():
            (output / "smoke-failure.json").write_text(json.dumps({"passed": False, "error": str(exc),
                                                                 "checks": checks}, indent=2), encoding="utf-8")
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Create a small source ZIP and audit wheel/sdist/ZIP payloads before release."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT_FILES = {"README.md", "README.ru.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE", "pyproject.toml", "MANIFEST.in", ".gitignore", ".gitattributes",
              "requirements-tested-cpu.txt", "requirements-tested-cuda.txt", "requirements-tested-gguf.txt"}
DIRECTORIES = {"kernelixx": {".py"}, "examples": {".py"}, "tests": {".py"}, "tools": {".py"},
               "docs": {".md", ".json", ".csv", ".txt"}, ".github": {".yml", ".yaml"}}
FORBIDDEN_PARTS = {".venv", "results", "build", "dist", ".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".onnx", ".gguf", ".npz", ".npy", ".pkl", ".jpg", ".jpeg", ".png",
                      ".dll", ".pyd", ".pyc", ".pyo"}


def source_files(root: Path) -> list[Path]:
    selected = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory, suffixes in DIRECTORIES.items():
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and not (set(path.relative_to(root).parts) & FORBIDDEN_PARTS):
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Source points outside the repository: {path}")
                selected.append(path)
    return sorted(selected)


def audit(path: Path, root: Path, version: str) -> dict[str, Any]:
    if path.suffix == ".gz":
        with tarfile.open(path, "r:gz") as archive:
            names = [member.name for member in archive.getmembers() if member.isfile()]
    else:
        with zipfile.ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
    for name in names:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts or set(member.parts) & FORBIDDEN_PARTS:
            raise ValueError(f"Unexpected archive path: {name}")
        if member.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Generated/binary data accidentally packaged: {name}")
    core = {path.relative_to(root).as_posix() for path in source_files(root) if path.is_relative_to(root / "kernelixx")}
    if path.suffix == ".whl":
        expected = core | {f"kernelixx-{version}.dist-info/METADATA",
                           f"kernelixx-{version}.dist-info/entry_points.txt",
                           f"kernelixx-{version}.dist-info/licenses/LICENSE"}
        normalized = set(names)
    else:
        normalized = {str(PurePosixPath(*PurePosixPath(name).parts[1:])) for name in names}
        expected = {item.relative_to(root).as_posix() for item in source_files(root)}
    missing = expected - normalized
    if missing:
        raise ValueError(f"{path.name} is missing source files: {sorted(missing)}")
    # Verify actual source bytes as well as names, so stale build outputs fail.
    checked = core if path.suffix == ".whl" else expected
    if path.suffix == ".gz":
        with tarfile.open(path, "r:gz") as archive:
            lookup = {str(PurePosixPath(*PurePosixPath(member.name).parts[1:])): member
                      for member in archive.getmembers() if member.isfile()}
            for name in checked:
                member = archive.extractfile(lookup[name])
                if member is None or member.read() != (root / name).read_bytes():
                    raise ValueError(f"Stale source in {path.name}: {name}")
    else:
        with zipfile.ZipFile(path) as archive:
            lookup = {name if path.suffix == ".whl" else str(PurePosixPath(*PurePosixPath(name).parts[1:])): name
                      for name in names}
            for name in checked:
                if archive.read(lookup[name]) != (root / name).read_bytes():
                    raise ValueError(f"Stale source in {path.name}: {name}")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"filename": path.name, "size_bytes": path.stat().st_size, "file_count": len(names),
            "sha256": digest, "source_content_verified": True, "files": names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace the source ZIP for this version")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "dist"
    try:
        with (root / "pyproject.toml").open("rb") as source:
            version = tomllib.load(source)["project"]["version"]
        output.mkdir(parents=True, exist_ok=True)
        wheel = output / f"kernelixx-{version}-py3-none-any.whl"
        sdist = output / f"kernelixx-{version}.tar.gz"
        if not wheel.is_file() or not sdist.is_file():
            raise FileNotFoundError("Build wheel and source distribution first: python -m build")
        bundle = output / f"Kernelixx-{version}-source.zip"
        with zipfile.ZipFile(bundle, "w" if args.overwrite else "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in source_files(root):
                archive.write(path, f"Kernelixx-{version}/{path.relative_to(root).as_posix()}")
        records = [audit(path, root, version) for path in (wheel, sdist, bundle)]
        (output / "release-manifest.json").write_text(json.dumps({"version": version, "artifacts": records}, indent=2), encoding="utf-8")
        (output / "SHA256SUMS.txt").write_text("".join(f"{record['sha256']}  {record['filename']}\n" for record in records), encoding="utf-8")
        for record in records:
            print(f"Audited {record['filename']}: {record['size_bytes']} bytes, {record['file_count']} files")
        print("Source excludes .venv/, results/, model binaries and image arrays; metric reports remain in docs/.")
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Repeatable batch streams with bounded resident data and explicit disk spooling."""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


class BatchSource:
    def __init__(self, config: Any):
        self.config = config
        self.temporary = None
        self.arrays: list[np.ndarray] = []

    def __enter__(self) -> BatchSource:
        try:
            if self.config.dataset:
                path = Path(self.config.dataset)
                if path.is_dir():
                    directory = path
                else:
                    # NPZ has no mmap support. Extract only two known NPY members,
                    # using fixed target names (never archive-controlled paths).
                    self.temporary = tempfile.TemporaryDirectory(prefix="kernelixx-data-")
                    directory = Path(self.temporary.name)
                    with zipfile.ZipFile(path) as archive:
                        for name in ("inputs.npy", "labels.npy"):
                            if archive.namelist().count(name) != 1:
                                raise ValueError(f"NPZ requires exactly one {name}")
                            with archive.open(name) as source, (directory / name).open("wb") as target:
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                for name in ("inputs.npy", "labels.npy"):
                    self.arrays.append(np.load(directory / name, mmap_mode="r", allow_pickle=False))
                x, y = self.arrays
                if x.ndim != len(self.config.input_shape) or tuple(x.shape[1:]) != self.config.input_shape[1:]:
                    raise ValueError("Dataset feature shape must match --input-shape")
                if x.dtype != np.float32 or y.dtype != np.int64 or y.shape != (len(x),) or not len(x):
                    raise ValueError("Dataset requires float32 inputs and int64 labels [N], N > 0")
            return self
        except BaseException:
            self.__exit__()
            raise

    def __len__(self) -> int:
        return ((len(self.arrays[0]) + self.config.input_shape[0] - 1) // self.config.input_shape[0]
                if self.arrays else self.config.validation_batches)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        if not self.arrays:
            generator = torch.Generator().manual_seed(self.config.seed + 1)
            for _ in range(self.config.validation_batches):
                yield torch.randn(self.config.input_shape, generator=generator), None
        else:
            x, y = self.arrays
            size = self.config.input_shape[0]
            for index in range(0, len(x), size):
                chunk = self._copy_slice(x, index, size)
                if not np.isfinite(chunk).all():
                    raise ValueError("Dataset inputs must be finite")
                yield torch.from_numpy(chunk), torch.from_numpy(self._copy_slice(y, index, size))

    @staticmethod
    def _copy_slice(array: np.memmap, index: int, size: int) -> np.ndarray:
        # Unmap each consumed region. Keeping one whole-dataset map live allows
        # visited pages to accumulate in the Windows process working set.
        if array.flags.c_contiguous:
            shape = (min(size, len(array) - index), *array.shape[1:])
            stride = array.dtype.itemsize * int(np.prod(array.shape[1:], dtype=np.int64))
            region = np.memmap(array.filename, mode="r", dtype=array.dtype,
                               offset=array.offset + index * stride, shape=shape)
            try:
                return np.array(region, copy=True)
            finally:
                region._mmap.close()
        region = np.load(array.filename, mmap_mode="r", allow_pickle=False)
        try:
            return np.array(region[index:index + size], copy=True, order="C")
        finally:
            region._mmap.close()

    def __exit__(self, *args: Any) -> None:
        for array in self.arrays:
            array._mmap.close()
        self.arrays.clear()
        if self.temporary is not None:
            self.temporary.cleanup()


def dataset_sha256(path: str) -> str:
    import hashlib
    source = Path(path)
    if source.is_file():
        with source.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    digest = hashlib.sha256()
    for name in ("inputs.npy", "labels.npy"):
        digest.update(name.encode())
        with (source / name).open("rb") as stream:
            digest.update(bytes.fromhex(hashlib.file_digest(stream, "sha256").hexdigest()))
    return digest.hexdigest()

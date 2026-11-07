"""Scoped observations: sampled RSS is not an exact peak or model-only memory."""
from __future__ import annotations

import os
import sys
import threading
from typing import Any

import torch


class MemoryMonitor:
    def __init__(self, device: torch.device | None = None, interval: float = 0.01):
        if interval <= 0:
            raise ValueError("Memory sampling interval must be positive")
        self.device = device or torch.device("cpu")
        self.interval = interval
        self.stop = threading.Event()
        self.process = None
        self.nvml = None
        self.handle = None
        self.rss_before = self.rss_after = self.rss_peak = None
        self.vram_before = self.vram_after = self.vram_peak = None
        self.notes: list[str] = []

    def _vram(self) -> int | None:
        if self.handle is None:
            return None
        try:
            entries = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            values = [p.usedGpuMemory for p in entries if p.pid == os.getpid()]
            # WDDM often returns NVML_VALUE_NOT_AVAILABLE (an unsigned sentinel).
            if not values or any(v is None or v < 0 or v >= 2**63 for v in values):
                return None
            return sum(values)
        except Exception:
            return None

    def _sample(self) -> None:
        if self.process is not None:
            try:
                self.rss_after = self.process.memory_info().rss
                self.rss_peak = max(self.rss_peak or 0, self.rss_after)
            except OSError as exc:
                self.notes.append(f"RSS sample unavailable: {exc}")
        value = self._vram()
        if value is not None:
            self.vram_after = value
            self.vram_peak = max(self.vram_peak or 0, value)

    def __enter__(self) -> MemoryMonitor:
        try:
            import psutil
            self.process = psutil.Process()
            self.rss_before = self.rss_peak = self.process.memory_info().rss
        except (ImportError, OSError) as exc:
            self.notes.append(f"RSS unavailable; install kernelixx[memory]: {exc}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.allocated_before = torch.cuda.memory_allocated(self.device)
            self.reserved_before = torch.cuda.memory_reserved(self.device)
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                uuid = getattr(torch.cuda.get_device_properties(self.device), "uuid", None)
                if uuid is None:
                    raise RuntimeError("Cannot map CUDA device to NVML UUID")
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
                self.vram_before = self._vram()
            except Exception as exc:
                self.notes.append(f"NVML process VRAM unavailable: {exc}")
        self._sample()

        def sample() -> None:
            while not self.stop.wait(self.interval):
                self._sample()

        self.thread = threading.Thread(target=sample, daemon=True, name="kernelixx-memory")
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            self.stop.set()
            self.thread.join()
            self._sample()
            lifetime = None
            if self.process is not None:
                info = self.process.memory_info()
                lifetime = getattr(info, "peak_wset", getattr(info, "peak_rss", None))
                if lifetime is None and sys.platform != "win32":
                    import resource
                    lifetime = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
            cuda = self.device.type == "cuda"
            self.metrics = {
                "process_rss_before_bytes": self.rss_before,
                "process_rss_current_bytes": self.rss_after,
                "sampled_process_rss_peak_bytes": self.rss_peak,
                "process_rss_delta_bytes": self.rss_after - self.rss_before if self.rss_after is not None else None,
                "os_process_lifetime_peak_rss_bytes": lifetime,
                "sampling_interval_seconds": self.interval,
                "torch_allocated_before_bytes": self.allocated_before if cuda else None,
                "torch_allocated_current_bytes": torch.cuda.memory_allocated(self.device) if cuda else None,
                "torch_reserved_before_bytes": self.reserved_before if cuda else None,
                "torch_reserved_current_bytes": torch.cuda.memory_reserved(self.device) if cuda else None,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if cuda else None,
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if cuda else None,
                "torch_transient_allocated_bytes": torch.cuda.max_memory_allocated(self.device) - self.allocated_before if cuda else None,
                "process_vram_before_bytes": self.vram_before,
                "process_vram_current_bytes": self.vram_after,
                "sampled_process_vram_peak_bytes": self.vram_peak,
                "notes": self.notes + ["Periodic sampling can miss short peaks; OS peak covers the process lifetime.",
                                        "PyTorch allocated/reserved memory is not total process VRAM."],
            }
            if self.nvml is not None:
                self.nvml.nvmlShutdown()

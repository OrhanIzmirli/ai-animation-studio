"""Available system RAM, without third-party dependencies.

Used to record how much memory margin a local run had. Returns None on
platforms where it cannot be read; callers treat that as "unknown".
"""

from __future__ import annotations

import ctypes
import sys
import threading

GIB = 2**30


def ram_bytes() -> tuple[int, int] | None:
    """(total, available) physical memory in bytes, or None if unknown."""
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        with open("/proc/meminfo", encoding="ascii") as f:
            fields = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f if ":" in line}
        return fields["MemTotal"], fields["MemAvailable"]
    except (OSError, KeyError, ValueError, AttributeError):
        return None


def available_ram_gb() -> float | None:
    memory = ram_bytes()
    return None if memory is None else memory[1] / GIB


def total_ram_gb() -> float | None:
    memory = ram_bytes()
    return None if memory is None else memory[0] / GIB


class LowestAvailableRam:
    """Context manager that samples available RAM in a background thread and
    keeps the lowest value seen (GiB), or None if it cannot be read."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.lowest_gb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        value = available_ram_gb()
        if value is not None and (self.lowest_gb is None or value < self.lowest_gb):
            self.lowest_gb = value

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> LowestAvailableRam:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._sample()

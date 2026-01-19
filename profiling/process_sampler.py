"""
Process memory sampling utilities (RSS/VMS) for macOS Apple Silicon experiments.

We use process RSS (resident set size) as a practical proxy for "VRAM utilization"
on Apple Silicon's unified memory architecture.

This module provides:
- A background sampler that records timestamped RSS/VMS at a fixed interval
- Helpers to compute baseline (median) and peak metrics over time windows
"""

from __future__ import annotations

import csv
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import psutil


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass(frozen=True)
class MemSample:
    timestamp: str
    pid: int
    rss_mb: float
    vms_mb: float
    elapsed_s: float  # seconds since sampler start (monotonic)


class ProcessSampler:
    """
    Sample memory of a PID in a background thread.

    - Stops automatically if the process exits.
    - Thread-safe: samples list can be read after join().
    """

    def __init__(
        self,
        pid: int,
        interval_s: float = 0.05,
        label: str = "",
        *,
        include_children: bool = False,
    ) -> None:
        self.pid = int(pid)
        self.interval_s = float(interval_s)
        self.label = label
        self.include_children = bool(include_children)

        self._t0 = None  # monotonic start
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[MemSample] = []

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Sampler already started")
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"ProcessSampler(pid={self.pid})", daemon=True)
        self._thread.start()

    @property
    def start_monotonic(self) -> Optional[float]:
        return self._t0

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return

        while not self._stop.is_set():
            try:
                # Root process memory
                rss = 0
                vms = 0

                procs = [proc]
                if self.include_children:
                    try:
                        procs.extend(proc.children(recursive=True))
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        procs = [proc]

                for p in procs:
                    try:
                        mem = p.memory_info()
                        rss += mem.rss
                        vms += mem.vms
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue

                t = time.monotonic()
                assert self._t0 is not None
                self.samples.append(
                    MemSample(
                        timestamp=_now_iso(),
                        pid=self.pid,
                        rss_mb=rss / (1024 * 1024),
                        vms_mb=vms / (1024 * 1024),
                        elapsed_s=t - self._t0,
                    )
                )
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            except psutil.AccessDenied:
                # If access is denied, stop sampling to avoid a tight loop.
                break

            time.sleep(self.interval_s)

    def write_csv(self, path: str) -> None:
        """
        Write samples to CSV. Includes elapsed_s for easier windowing.
        """
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["timestamp", "label", "pid", "rss_mb", "vms_mb", "elapsed_s"],
            )
            w.writeheader()
            for s in self.samples:
                w.writerow(
                    {
                        "timestamp": s.timestamp,
                        "label": self.label,
                        "pid": s.pid,
                        "rss_mb": f"{s.rss_mb:.6f}",
                        "vms_mb": f"{s.vms_mb:.6f}",
                        "elapsed_s": f"{s.elapsed_s:.6f}",
                    }
                )


def sample_pid_for_duration(pid: int, duration_s: float, interval_s: float, label: str = "") -> List[MemSample]:
    sampler = ProcessSampler(pid=pid, interval_s=interval_s, label=label)
    sampler.start()
    time.sleep(duration_s)
    sampler.stop()
    sampler.join(timeout=5)
    return sampler.samples


def median_rss_mb(samples: Sequence[MemSample]) -> Optional[float]:
    if not samples:
        return None
    return statistics.median([s.rss_mb for s in samples])


def peak_rss_mb(samples: Sequence[MemSample]) -> Optional[float]:
    if not samples:
        return None
    return max(s.rss_mb for s in samples)


def filter_by_elapsed(samples: Sequence[MemSample], start_s: float, end_s: float) -> List[MemSample]:
    """
    Return samples whose elapsed_s is in [start_s, end_s].
    """
    return [s for s in samples if start_s <= s.elapsed_s <= end_s]


def compute_window_metrics(
    samples: Sequence[MemSample],
    *,
    baseline_window: Tuple[float, float],
    prefill_window: Tuple[float, float],
    total_window: Tuple[float, float],
) -> dict:
    """
    Compute baseline (median) and peak RSS for baseline/prefill/total windows.
    """
    b = filter_by_elapsed(samples, *baseline_window)
    p = filter_by_elapsed(samples, *prefill_window)
    t = filter_by_elapsed(samples, *total_window)

    return {
        "baseline_idle_rss_mb": median_rss_mb(b),
        "peak_prefill_rss_mb": peak_rss_mb(p),
        "peak_total_rss_mb": peak_rss_mb(t),
        "baseline_window_s": baseline_window,
        "prefill_window_s": prefill_window,
        "total_window_s": total_window,
        "sample_count": len(samples),
    }


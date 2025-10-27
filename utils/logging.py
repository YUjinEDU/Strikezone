"""Runtime logging helpers for strike-zone processing."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional


@dataclass
class ExponentialMovingAverage:
    """Maintain an exponential moving average (EMA)."""

    alpha: float
    value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None:
            self.value = sample
        else:
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class PerformanceSnapshot:
    """Snapshot of the smoothed performance metrics."""

    fps: float
    latency_ms: float
    dropped_total: int
    dropped_recent: int

    def format(self) -> str:
        return (
            f"FPS(avg): {self.fps:5.2f} | latency(avg): {self.latency_ms:6.1f} ms | "
            f"dropped(total/recent): {self.dropped_total}/{self.dropped_recent}"
        )


class PerformanceLogger:
    """Track FPS, latency and dropped frames using exponential smoothing."""

    def __init__(
        self,
        fps_alpha: float = 0.2,
        latency_alpha: float = 0.2,
    ) -> None:
        self._fps_ema = ExponentialMovingAverage(fps_alpha)
        self._latency_ema = ExponentialMovingAverage(latency_alpha)
        self._last_capture_ts: Optional[float] = None
        self._last_frame_index: Optional[int] = None
        self._dropped_total = 0

    def update(
        self,
        frame_index: int,
        capture_timestamp: float,
        processed_timestamp: Optional[float] = None,
        dropped: bool = False,
    ) -> PerformanceSnapshot:
        now = processed_timestamp if processed_timestamp is not None else time.time()

        fps_sample = 0.0
        if self._last_capture_ts is not None:
            dt = capture_timestamp - self._last_capture_ts
            if dt > 0:
                fps_sample = 1.0 / dt
        fps_avg = self._fps_ema.update(fps_sample)

        latency = max(now - capture_timestamp, 0.0)
        latency_avg = self._latency_ema.update(latency)

        dropped_recent = 0
        if self._last_frame_index is not None:
            gap = frame_index - self._last_frame_index
            if gap > 1:
                dropped_recent += gap - 1
        if dropped:
            dropped_recent += 1
        self._dropped_total += dropped_recent

        self._last_capture_ts = capture_timestamp
        self._last_frame_index = frame_index

        return PerformanceSnapshot(
            fps=fps_avg,
            latency_ms=latency_avg * 1000.0,
            dropped_total=self._dropped_total,
            dropped_recent=dropped_recent,
        )

    def reset(self) -> None:
        self._fps_ema.reset()
        self._latency_ema.reset()
        self._last_capture_ts = None
        self._last_frame_index = None
        self._dropped_total = 0


__all__ = [
    "ExponentialMovingAverage",
    "PerformanceLogger",
    "PerformanceSnapshot",
]

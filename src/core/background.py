"""Background modelling constrained to the region of interest."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from utils.logging import get_logger


@dataclass
class BackgroundModel:
    """Maintain a rolling-median background and perform motion checks."""

    window: int = 5
    freeze_frames: int = 3

    def __post_init__(self) -> None:
        self._logger = get_logger("background")
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.window)
        self._median: Optional[np.ndarray] = None
        self._freeze_count = 0

    def update(self, frame_roi: np.ndarray) -> None:
        """Update the background model with a new ROI frame.

        Args:
            frame_roi: ROI-cropped frame used to refresh the median buffer.
        """

        if self._freeze_count > 0:
            self._freeze_count -= 1
            return

        frame_f = frame_roi.astype(np.float32)
        self._buffer.append(frame_f)
        if len(self._buffer) == self.window:
            self._median = np.median(np.stack(self._buffer, axis=0), axis=0)

    def diff(self, frame_roi: np.ndarray) -> np.ndarray:
        """Return the absolute difference between the frame and the background.

        Args:
            frame_roi: ROI-cropped frame to compare against the background.

        Returns:
            Float32 absolute difference image.
        """

        if self._median is None:
            return np.zeros_like(frame_roi, dtype=np.float32)
        return np.abs(frame_roi.astype(np.float32) - self._median.astype(np.float32))

    def detect_motion(self, frame_roi: np.ndarray, percentile: float = 90.0, threshold: float = 15.0) -> bool:
        """Determine whether motion is present in the ROI.

        Args:
            frame_roi: ROI-cropped frame used for motion evaluation.
            percentile: Percentile of the difference image used as activity score.
            threshold: Threshold above which motion is flagged.

        Returns:
            ``True`` if motion is detected, ``False`` otherwise.
        """

        if self._median is None:
            return False
        delta = self.diff(frame_roi)
        score = np.percentile(delta, percentile)
        motion = score > threshold
        if motion:
            self._freeze_count = self.freeze_frames
        return motion


__all__ = ["BackgroundModel"]

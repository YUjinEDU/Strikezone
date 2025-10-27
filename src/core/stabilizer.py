"""Frame stabilization utilities leveraging sparse optical flow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class StabilizationResult:
    """Result for a stabilized frame."""

    frame: np.ndarray
    transform: np.ndarray
    translation: Tuple[float, float]


class FrameStabilizer:
    """Stabilize frames by estimating inter-frame translation within an ROI."""

    def __init__(self, frame_shape: Tuple[int, int, int]) -> None:
        self.height, self.width = frame_shape[:2]
        self.prev_gray: Optional[np.ndarray] = None
        self.transform = np.eye(2, 3, dtype=np.float32)

    def reset(self) -> None:
        """Forget the previous frame state."""
        self.prev_gray = None
        self.transform = np.eye(2, 3, dtype=np.float32)

    def stabilize(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> StabilizationResult:
        """Stabilize the incoming frame using sparse optical flow in ROI.

        Parameters
        ----------
        frame:
            BGR frame to stabilize.
        roi:
            ROI defined as ``(x, y, w, h)``.

        Returns
        -------
        StabilizationResult
            Stabilized frame and transformation information.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = roi
        roi_gray = gray[y : y + h, x : x + w]

        if self.prev_gray is None:
            self.prev_gray = gray
            return StabilizationResult(frame=frame, transform=self.transform.copy(), translation=(0.0, 0.0))

        prev_roi = self.prev_gray[y : y + h, x : x + w]
        features = cv2.goodFeaturesToTrack(prev_roi, maxCorners=50, qualityLevel=0.01, minDistance=5)
        if features is None:
            LOGGER.debug("No features detected for stabilization; skipping update")
            self.prev_gray = gray
            return StabilizationResult(frame=frame, transform=self.transform.copy(), translation=(0.0, 0.0))

        features += np.array([[x, y]], dtype=np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, features, None)
        valid = status.squeeze() == 1
        if not np.any(valid):
            LOGGER.debug("Optical flow failed; keeping previous transform")
            self.prev_gray = gray
            return StabilizationResult(frame=frame, transform=self.transform.copy(), translation=(0.0, 0.0))

        prev_pts = features[valid]
        next_pts_valid = next_pts[valid]
        translation = next_pts_valid - prev_pts
        dx, dy = np.median(translation, axis=0)
        self.transform = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)

        stabilized = cv2.warpAffine(frame, self.transform, (self.width, self.height), flags=cv2.INTER_LINEAR)
        self.prev_gray = gray
        return StabilizationResult(frame=stabilized, transform=self.transform.copy(), translation=(float(dx), float(dy)))

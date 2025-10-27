"""Background modelling for motion segmentation."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

class BackgroundModel:
    """Simple running-average background model constrained to ROI."""

    def __init__(self, frame_shape: Tuple[int, int, int], alpha: float = 0.02) -> None:
        self.height, self.width = frame_shape[:2]
        self.alpha = alpha
        self.background = np.zeros((self.height, self.width), dtype=np.float32)
        self.diff = np.zeros((self.height, self.width), dtype=np.uint8)
        self.initialized = False

    def apply(self, frame_gray: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Update background and return absolute difference within ROI."""
        x, y, w, h = roi
        roi_gray = frame_gray[y : y + h, x : x + w]
        roi_bg = self.background[y : y + h, x : x + w]
        roi_diff = self.diff[y : y + h, x : x + w]
        if not self.initialized:
            roi_bg[:] = roi_gray
            self.initialized = True
        else:
            cv2.accumulateWeighted(roi_gray, roi_bg, self.alpha)
        cv2.absdiff(roi_gray, cv2.convertScaleAbs(roi_bg, dst=roi_diff), dst=roi_diff)
        return roi_diff

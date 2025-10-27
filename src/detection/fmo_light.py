"""Fast moving object (FMO) detection tailored for baseballs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class Candidate:
    """Represents a detected fast moving object candidate."""

    center: Tuple[float, float]
    dt_peak: float
    diameter_px: float
    mask: np.ndarray


class FmoLightDetector:
    """Lightweight detector using background differencing and DT peaks."""

    def __init__(self, roi_shape: Tuple[int, int], threshold_factor: float = 0.45) -> None:
        self.height, self.width = roi_shape
        self.threshold_factor = threshold_factor
        self.binary = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dt = np.zeros((self.height, self.width), dtype=np.float32)

    def _update_buffers(self, roi_shape: Tuple[int, int]) -> None:
        if roi_shape != (self.height, self.width):
            self.height, self.width = roi_shape
            self.binary = np.zeros((self.height, self.width), dtype=np.uint8)
            self.dt = np.zeros((self.height, self.width), dtype=np.float32)

    def detect(self, diff: np.ndarray, roi: Tuple[int, int, int, int], color_mask: Optional[np.ndarray] = None) -> Optional[Candidate]:
        """Detect a fast moving object within the ROI.

        Parameters
        ----------
        diff:
            Grayscale difference image restricted to ROI.
        roi:
            ROI coordinates relative to the full frame.
        color_mask:
            Optional binary mask representing color prior intersection.

        Returns
        -------
        Optional[Candidate]
            Detected candidate if any.
        """

        h, w = diff.shape
        self._update_buffers((h, w))
        mean_intensity = float(np.mean(diff))
        thresh = max(15, mean_intensity * (1.0 + self.threshold_factor))
        cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY, dst=self.binary)
        if color_mask is not None:
            cv2.bitwise_and(self.binary, color_mask, dst=self.binary)

        if not np.any(self.binary):
            return None

        cv2.distanceTransform(self.binary, cv2.DIST_L2, 3, dst=self.dt)
        _, max_val, _, max_loc = cv2.minMaxLoc(self.dt)
        if max_val <= 0:
            return None

        diameter_px = max_val * 2
        candidate_mask = self.binary.copy()
        center = (roi[0] + max_loc[0], roi[1] + max_loc[1])
        LOGGER.debug("Candidate found at %s with diameter %.2f px", center, diameter_px)
        return Candidate(center=center, dt_peak=float(max_val), diameter_px=float(diameter_px), mask=candidate_mask)

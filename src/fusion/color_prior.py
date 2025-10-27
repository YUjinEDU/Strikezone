"""Color prior fusion for baseball detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class ColorPriorConfig:
    """Configuration for the color prior."""

    lower_hsv: Tuple[int, int, int] = (0, 0, 180)
    upper_hsv: Tuple[int, int, int] = (30, 80, 255)


class ColorPrior:
    """Compute binary masks based on configured color range."""

    def __init__(self, config: ColorPriorConfig) -> None:
        self.config = config

    def compute(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Return a binary mask of pixels consistent with the color prior."""
        x, y, w, h = roi
        hsv = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.lower_hsv, self.config.upper_hsv)
        return mask

"""ROI management utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ROIState:
    """Represents an ROI box."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the ROI as tuple."""
        return self.x, self.y, self.w, self.h


class ROIGate:
    """Simple ROI mode manager maintaining ROI within frame bounds."""

    def __init__(self, frame_shape: Tuple[int, int], mode: str = "expanded", expand_ratio: float = 0.2) -> None:
        self.height, self.width = frame_shape
        self.mode = mode
        self.expand_ratio = expand_ratio
        self.manual_roi = ROIState(0, 0, self.width, self.height)

    def cycle_mode(self) -> str:
        """Cycle ROI mode in order expanded → tight → manual."""
        order = ["expanded", "tight", "manual"]
        idx = order.index(self.mode)
        self.mode = order[(idx + 1) % len(order)]
        return self.mode

    def update_manual_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """Update manual ROI value."""
        x, y, w, h = roi
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        w = int(np.clip(w, 1, self.width - x))
        h = int(np.clip(h, 1, self.height - y))
        self.manual_roi = ROIState(x, y, w, h)

    def compute_roi(self, base_roi: ROIState) -> ROIState:
        """Compute ROI given base detection ROI."""
        if self.mode == "manual":
            return self.manual_roi
        if self.mode == "tight":
            return base_roi
        # expanded
        expand_w = int(base_roi.w * self.expand_ratio)
        expand_h = int(base_roi.h * self.expand_ratio)
        x = max(base_roi.x - expand_w, 0)
        y = max(base_roi.y - expand_h, 0)
        w = min(base_roi.w + 2 * expand_w, self.width - x)
        h = min(base_roi.h + 2 * expand_h, self.height - y)
        return ROIState(x, y, w, h)

    def full_frame(self) -> ROIState:
        """Return ROI covering entire frame."""
        return ROIState(0, 0, self.width, self.height)

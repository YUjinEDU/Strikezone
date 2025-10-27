"""Camera parameter utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CameraParameters:
    """Simplified pinhole camera parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        """Return the camera intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def enforce_aspect(params: CameraParameters, frame_shape: Tuple[int, int]) -> CameraParameters:
    """Return a copy of parameters ensuring aspect ratio consistency."""
    height, width = frame_shape
    scale_x = width / (2 * params.cx)
    scale_y = height / (2 * params.cy)
    scale = (scale_x + scale_y) / 2
    return CameraParameters(fx=params.fx * scale, fy=params.fy * scale, cx=params.cx, cy=params.cy)

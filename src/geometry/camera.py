"""Camera model helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass
class CameraModel:
    """Simple pinhole camera model with distortion."""

    matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

    def project(self, points: Iterable[Iterable[float]]) -> np.ndarray:
        """Project 3D points into image space.

        Args:
            points: Iterable of 3D coordinates expressed in the world frame.

        Returns:
            Array of projected pixel coordinates with shape ``(N, 2)``.
        """

        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 3)
        img_pts, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.matrix, self.dist_coeffs)
        return img_pts.reshape(-1, 2)


__all__ = ["CameraModel"]

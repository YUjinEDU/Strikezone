"""Depth projection helpers."""

from __future__ import annotations

import numpy as np

from .config import CameraIntrinsics


def back_project_uvz_to_xyz(u: float, v: float, z: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project image coordinates and depth to 3-D camera coordinates."""

    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=float)


def project_xyz_to_uv(point: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project a 3-D camera point into pixel coordinates."""

    if point.shape != (3,):
        raise ValueError("A single 3-D point is required for projection.")

    x, y, z = point
    if z <= 0:
        raise ValueError("Points must be in front of the camera (z > 0).")

    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    return np.array([u, v], dtype=float)

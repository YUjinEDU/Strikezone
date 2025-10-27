"""Camera geometry helpers used by the FMO tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class CameraExtrinsics:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if self.rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")


@dataclass(slots=True)
class CameraModel:
    intr: CameraIntrinsics
    extr: CameraExtrinsics

    @property
    def fx(self) -> float:
        return self.intr.fx

    @property
    def fy(self) -> float:
        return self.intr.fy

    @property
    def cx(self) -> float:
        return self.intr.cx

    @property
    def cy(self) -> float:
        return self.intr.cy

    @property
    def rotation(self) -> np.ndarray:
        return self.extr.rotation

    @property
    def translation(self) -> np.ndarray:
        return self.extr.translation


def depth_from_diameter(d_px: float, ball_diameter_m: float, intr: CameraIntrinsics) -> float:
    """Estimate the depth using the observed diameter in pixels."""

    if d_px <= 0:
        raise ValueError("d_px must be positive")
    return intr.fx * ball_diameter_m / float(d_px)


def image_to_camera(uv: Tuple[float, float], depth: float, intr: CameraIntrinsics) -> np.ndarray:
    """Back-project an image point into the camera reference frame."""

    u, v = uv
    x = (u - intr.cx) * depth / intr.fx
    y = (v - intr.cy) * depth / intr.fy
    return np.array([x, y, depth], dtype=np.float64)


def camera_to_world(point_cam: np.ndarray, extr: CameraExtrinsics) -> np.ndarray:
    """Transform camera coordinates to world coordinates."""

    return extr.rotation @ point_cam + extr.translation


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraModel",
    "camera_to_world",
    "depth_from_diameter",
    "image_to_camera",
]

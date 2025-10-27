"""Mapping between image measurements and world coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .camera import CameraParameters


@dataclass
class TrajectoryEstimate:
    """Estimated 3D position and velocity."""

    position_m: np.ndarray
    velocity_mps: np.ndarray


def diameter_from_dt_peak(dt_peak: float) -> float:
    """Return approximate diameter in pixels from DT peak."""
    return dt_peak * 2.0


def depth_from_diameter(params: CameraParameters, real_diameter_m: float, diameter_px: float) -> float:
    """Compute depth from pixel diameter using pinhole model."""
    if diameter_px <= 0:
        raise ValueError("Diameter must be positive")
    return params.fx * real_diameter_m / diameter_px


def pixel_displacement_to_velocity(
    displacement_px: Tuple[float, float],
    params: CameraParameters,
    depth_m: float,
    frame_time: float,
) -> np.ndarray:
    """Convert pixel displacement to velocity in meters per second."""
    dx, dy = displacement_px
    if frame_time <= 0:
        raise ValueError("Frame time must be positive")
    scale = depth_m / params.fx
    vx = dx * scale / frame_time
    vy = dy * scale / frame_time
    return np.array([vx, vy, 0.0], dtype=np.float32)


def triangulate_position(center_px: Tuple[float, float], depth_m: float, params: CameraParameters) -> np.ndarray:
    """Triangulate the 3D point from pixel coordinates and depth."""
    x = (center_px[0] - params.cx) * depth_m / params.fx
    y = (center_px[1] - params.cy) * depth_m / params.fy
    return np.array([x, y, depth_m], dtype=np.float32)


def estimate_trajectory(
    center_px: Tuple[float, float],
    displacement_px: Tuple[float, float],
    params: CameraParameters,
    real_diameter_m: float,
    diameter_px: float,
    frame_time: float,
) -> TrajectoryEstimate:
    """Estimate 3D trajectory information from pixel measurements."""
    depth = depth_from_diameter(params, real_diameter_m, diameter_px)
    position = triangulate_position(center_px, depth, params)
    velocity = pixel_displacement_to_velocity(displacement_px, params, depth, frame_time)
    return TrajectoryEstimate(position_m=position, velocity_mps=velocity)

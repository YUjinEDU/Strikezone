"""Velocity estimation validation using synthetic sequence."""
from __future__ import annotations

import numpy as np

from src.geometry.camera import CameraParameters
from src.geometry.mapping import pixel_displacement_to_velocity


def test_velocity_with_synthetic_motion() -> None:
    params = CameraParameters(fx=900.0, fy=900.0, cx=320.0, cy=240.0)
    depth = 18.0  # meters
    frame_time = 1.0 / 60.0
    true_velocity = np.array([40.0, -20.0, 0.0])
    scale = params.fx / depth
    displacement_px = (true_velocity[0] * frame_time * scale, true_velocity[1] * frame_time * scale)
    estimate = pixel_displacement_to_velocity(displacement_px, params, depth, frame_time)
    error = np.linalg.norm(estimate[:2] - true_velocity[:2]) / np.linalg.norm(true_velocity[:2])
    assert error < 0.1

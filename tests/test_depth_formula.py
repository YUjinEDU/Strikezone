"""Tests for depth estimation formula."""
from __future__ import annotations

from src.geometry.camera import CameraParameters
from src.geometry.mapping import depth_from_diameter


def test_depth_formula() -> None:
    params = CameraParameters(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    real_diameter = 0.074
    diameter_px = 40.0
    depth = depth_from_diameter(params, real_diameter, diameter_px)
    expected = params.fx * real_diameter / diameter_px
    assert depth == expected

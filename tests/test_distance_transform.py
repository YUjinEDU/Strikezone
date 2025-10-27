"""Tests verifying the relation between DT peak and diameter."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.geometry.mapping import diameter_from_dt_peak


def test_dt_peak_matches_diameter() -> None:
    strip = np.zeros((50, 50), dtype=np.uint8)
    strip[20:30, 10:40] = 1
    dt = ndimage.distance_transform_edt(strip)
    dt_peak = float(dt.max())
    diameter_px = diameter_from_dt_peak(dt_peak)
    assert np.isclose(diameter_px, dt_peak * 2.0)

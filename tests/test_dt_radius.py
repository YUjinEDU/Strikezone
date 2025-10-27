import numpy as np

from strikezone.distance import estimate_radius_from_dt


def test_dt_peak_matches_half_width():
    height, width = 40, 40
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 15:25] = True  # streak 10 pixels wide

    radius = estimate_radius_from_dt(mask)

    assert np.isclose(radius, 5.0, atol=0.1)

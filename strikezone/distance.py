"""Utilities around distance transforms."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def estimate_radius_from_dt(binary_mask: np.ndarray) -> float:
    """Estimate the radius of a binary streak via the distance transform.

    Parameters
    ----------
    binary_mask:
        A 2-D boolean array where ``True`` denotes the streak interior.

    Returns
    -------
    float
        The peak value of the Euclidean distance transform, corresponding to
        the streak's half-width in pixels.
    """

    if binary_mask.ndim != 2:
        raise ValueError("The distance transform expects a 2-D mask.")

    mask = np.asarray(binary_mask, dtype=bool)
    if not np.any(mask):
        return 0.0

    distance = ndi.distance_transform_edt(mask)
    return float(distance.max())

"""Colour prior utilities for the fast moving object pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from detection.fmo_light import FmoCandidate


@dataclass(slots=True)
class HSVPreset:
    """HSV interval represented with inclusive lower/upper bounds."""

    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        lower_np = np.array(self.lower, dtype=np.uint8)
        upper_np = np.array(self.upper, dtype=np.uint8)
        return cv2.inRange(hsv, lower_np, upper_np) > 0


DEFAULT_WHITE_PRESET = HSVPreset(lower=(0, 0, 180), upper=(179, 60, 255))


def _prepare_presets(additional: Iterable[HSVPreset] | None) -> Tuple[HSVPreset, ...]:
    presets = [DEFAULT_WHITE_PRESET]
    if additional:
        presets.extend(additional)
    return tuple(presets)


def compute_color_score(
    frame_bgr: np.ndarray,
    roi: Tuple[int, int, int, int],
    candidate: FmoCandidate,
    additional_presets: Sequence[HSVPreset] | None = None,
) -> float:
    """Compute a colour-based likelihood score for a candidate.

    The score is the proportion of pixels inside the candidate mask that match
    at least one of the configured HSV presets.
    """

    x, y, w, h = roi
    roi_view = frame_bgr[y : y + h, x : x + w]
    hsv = cv2.cvtColor(roi_view, cv2.COLOR_BGR2HSV)

    mask = candidate.mask_roi.astype(bool)
    if mask.size == 0 or not mask.any():
        return 0.0

    presets = _prepare_presets(additional_presets)
    pooled = np.zeros_like(mask, dtype=bool)
    for preset in presets:
        pooled |= preset.mask(hsv)

    hits = np.logical_and(pooled, mask).sum()
    total = mask.sum()
    if total == 0:
        return 0.0
    return float(hits) / float(total)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def normalise_score(value: float, bias: float = 1.0) -> float:
    """Normalise a raw score into [0, 1) using a smooth squashing function."""

    if value <= 0:
        return 0.0
    return float(value / (value + bias))


def fuse_confidence(
    s_fmo: float,
    s_color: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Fuse FMO and colour scores into a final confidence."""

    norm = normalise_score(s_fmo)
    return sigmoid(alpha * norm + beta * s_color + gamma)


__all__ = [
    "HSVPreset",
    "compute_color_score",
    "fuse_confidence",
    "normalise_score",
    "sigmoid",
]

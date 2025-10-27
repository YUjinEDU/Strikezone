"""Lightweight fast-moving object (FMO) detector.

This module implements a simplified fast moving object detector based on
morphological processing and distance transforms.  The implementation follows
an internal design document used by the team; the most relevant steps are
summarised below so that callers do not need to reference external material:

* the input is a difference frame that isolates moving content inside a region
  of interest (ROI)
* the difference frame is converted to a binary mask using an adaptive
  threshold that depends on the frame statistics
* only large connected components survive the clean-up stage
* distance transforms provide an estimate of the streak half-width which is in
  turn used to infer the ball diameter in pixels
* each component is analysed with PCA to recover an axis aligned with the
  motion trail and to estimate its length/width
* heuristic scores are produced and the best candidates are returned

The detector is intentionally opinionated but small; the resulting candidates
are meant to be fused with additional cues (colour priors, temporal tracking,
3D mapping) by higher level code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(slots=True)
class FmoCandidate:
    """Container describing a single fast moving object detection candidate."""

    center_uv: Tuple[float, float]
    width_px: float
    length_px: float
    dt_peak: float
    axis_dir: Tuple[float, float]
    mask_roi: np.ndarray
    s_fmo: float
    bbox_roi: Tuple[int, int, int, int]


@dataclass(slots=True)
class DynamicFilterState:
    """Optional context used by the detector to prune unlikely streaks."""

    centerline_point: Tuple[float, float]
    centerline_dir: Tuple[float, float]
    expected_motion_dir: Optional[Tuple[float, float]] = None


@dataclass(slots=True)
class DetectorConfig:
    """Configuration knobs for :func:`detect_fmo_candidates`."""

    dfactor: float = 1.5
    gaussian_sigma: float = 0.0
    gaussian_ksize: int = 3
    min_component_area: int = 25
    max_width_px: float = 40.0
    min_speed_px_per_frame: float = 1.0
    max_aspect_ratio: float = 12.0
    min_aspect_ratio: float = 1.5
    aspect_slope: float = 2.0
    dynamic_centerline_sigma: float = 15.0
    dynamic_direction_weight: float = 0.3


def _extract_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return image[y : y + h, x : x + w]


def _binary_mask(diff_roi: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    gray = diff_roi
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (cfg.gaussian_ksize, cfg.gaussian_ksize), cfg.gaussian_sigma)
    mean, std = cv2.meanStdDev(blur)
    threshold = float(mean) + cfg.dfactor * float(std)
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _connected_components(mask: np.ndarray, cfg: DetectorConfig) -> Iterable[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < cfg.min_component_area:
            continue
        x, y, w, h, _ = stats[label]
        component_mask = (labels == label).astype(np.uint8)
        yield component_mask, (x, y, w, h)


def _distance_transform_stats(component_mask: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    dt = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    dt_peak = float(dt.max())
    dilated = cv2.dilate(dt, np.ones((3, 3), np.float32))
    maxima = (dt == dilated) & (dt > 0)
    return dt, dt_peak, maxima


def _pca_axis(component_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack(np.nonzero(component_mask))
    if points.shape[0] < 2:
        mean = points.mean(axis=0) if points.size else np.array([0.0, 0.0])
        cov = np.eye(2)
    else:
        mean = points.mean(axis=0)
        points_centered = points - mean
        cov = np.cov(points_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    minor_axis = eigvecs[:, 1]
    return mean, axis, minor_axis


def _length_width(points: np.ndarray, axis: np.ndarray, dt: np.ndarray, maxima: np.ndarray, cfg: DetectorConfig) -> Tuple[float, float]:
    if points.size == 0:
        return 0.0, 0.0
    projections = points @ axis
    length = float(projections.max() - projections.min())
    if np.any(maxima):
        dt_values = dt[maxima]
        width = float(np.mean(dt_values) * 2.0)
    else:
        width = float(dt.max() * 2.0)
    width = min(width, cfg.max_width_px)
    return length, width


def _centroid(component_mask: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(component_mask.astype(np.uint8))
    if m["m00"] == 0:
        ys, xs = np.nonzero(component_mask)
        if xs.size == 0:
            return 0.0, 0.0
        return float(xs.mean()), float(ys.mean())
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return float(cx), float(cy)


def _aspect_penalty(length: float, width: float, cfg: DetectorConfig) -> float:
    if width <= 0:
        return cfg.max_aspect_ratio
    aspect = length / max(width, 1e-6)
    penalty = 0.0
    if aspect > cfg.max_aspect_ratio:
        penalty += cfg.aspect_slope * (aspect - cfg.max_aspect_ratio)
    if aspect < cfg.min_aspect_ratio:
        penalty += cfg.aspect_slope * (cfg.min_aspect_ratio - aspect)
    return penalty


def _dynamic_filter(center: Tuple[float, float], axis: np.ndarray, state: DynamicFilterState, cfg: DetectorConfig) -> float:
    pt = np.asarray(state.centerline_point, dtype=np.float32)
    line_dir = np.asarray(state.centerline_dir, dtype=np.float32)
    line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-6)
    center_vec = np.asarray(center, dtype=np.float32)
    diff = center_vec - pt
    projected = diff - line_dir * float(np.dot(diff, line_dir))
    distance = np.linalg.norm(projected)
    spatial_score = float(np.exp(-(distance ** 2) / (2 * cfg.dynamic_centerline_sigma ** 2)))
    directional_score = 1.0
    if state.expected_motion_dir is not None:
        expected = np.asarray(state.expected_motion_dir, dtype=np.float32)
        expected = expected / (np.linalg.norm(expected) + 1e-6)
        axis_uv = np.array([axis[1], axis[0]], dtype=np.float32)
        axis_norm = axis_uv / (np.linalg.norm(axis_uv) + 1e-6)
        directional_score = max(0.0, float(np.dot(axis_norm, expected)))
        directional_score = (1 - cfg.dynamic_direction_weight) + cfg.dynamic_direction_weight * directional_score
    return spatial_score * directional_score


def detect_fmo_candidates(
    diff_frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    cfg: DetectorConfig | None = None,
    dynamic_state: DynamicFilterState | None = None,
    prev_centers: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[FmoCandidate]:
    """Run the FMO detector on the provided diff frame.

    Parameters
    ----------
    diff_frame:
        Difference (motion) frame computed by the caller.  Both grayscale and
        colour inputs are supported.
    roi:
        Region of interest (x, y, width, height).  Only data inside the ROI is
        used and coordinates are returned relative to the ROI origin.
    cfg:
        Optional detector configuration.  Defaults are tuned for high frame
        rate footage.
    dynamic_state:
        Optional hints coming from the tracker.  When provided, streaks close to
        the predicted centerline and aligned with the expected direction are
        favoured.
    prev_centers:
        Sequence with previously accepted centres.  Static objects that have not
        moved enough between frames are filtered out.
    """

    cfg = cfg or DetectorConfig()
    diff_roi = _extract_roi(diff_frame, roi)
    mask = _binary_mask(diff_roi, cfg)
    candidates: List[FmoCandidate] = []
    prev_centers = prev_centers or []

    for component_mask, bbox in _connected_components(mask, cfg):
        dt, dt_peak, maxima = _distance_transform_stats(component_mask)
        if dt_peak <= 0:
            continue
        points = np.column_stack(np.nonzero(component_mask))
        mean, axis, _ = _pca_axis(component_mask)
        length, width = _length_width(points, axis, dt, maxima, cfg)
        aspect_penalty = _aspect_penalty(length, width, cfg)
        score = (width * length) / (1.0 + aspect_penalty)

        center = _centroid(component_mask)
        global_center = (center[0] + bbox[0], center[1] + bbox[1])

        if prev_centers:
            distances = [np.hypot(global_center[0] - cx, global_center[1] - cy) for cx, cy in prev_centers]
            if distances and min(distances) < cfg.min_speed_px_per_frame:
                continue

        if dynamic_state is not None:
            score *= _dynamic_filter(global_center, axis, dynamic_state, cfg)

        if score <= 0:
            continue

        bbox_roi = bbox
        axis_dir = tuple(axis / (np.linalg.norm(axis) + 1e-6))
        candidates.append(
            FmoCandidate(
                center_uv=global_center,
                width_px=width,
                length_px=length,
                dt_peak=dt_peak,
                axis_dir=(float(axis_dir[1]), float(axis_dir[0])),
                mask_roi=component_mask.astype(bool),
                s_fmo=score,
                bbox_roi=bbox_roi,
            )
        )

    candidates.sort(key=lambda c: c.s_fmo, reverse=True)
    return candidates[:2]


__all__ = ["FmoCandidate", "DetectorConfig", "DynamicFilterState", "detect_fmo_candidates"]

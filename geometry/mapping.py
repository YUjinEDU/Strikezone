"""Mapping utilities turning 2D FMO candidates into 3D tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from detection.fmo_light import FmoCandidate

from .camera import CameraModel, depth_from_diameter, image_to_camera, camera_to_world


@dataclass(slots=True)
class FmoTrack:
    uv: Tuple[float, float]
    xyz: Tuple[float, float, float]
    width_px: float
    length_px: float
    z_from_diam: float
    v_mps: float
    conf: float
    axis_dir_2d: Tuple[float, float]
    bbox_roi: Tuple[int, int, int, int]


def _axis_unit(axis_dir_2d: Tuple[float, float]) -> np.ndarray:
    axis = np.asarray(axis_dir_2d, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.array([1.0, 0.0], dtype=np.float64)
    return axis / norm


def _exposure_velocity(
    uv: Tuple[float, float],
    axis_dir: Tuple[float, float],
    length_px: float,
    depth: float,
    camera: CameraModel,
    exposure_sec: float,
) -> float:
    if exposure_sec <= 0 or length_px <= 0:
        return 0.0
    axis = _axis_unit(axis_dir)
    displacement_uv = axis * length_px
    uv_end = (uv[0] + displacement_uv[0], uv[1] + displacement_uv[1])
    start_cam = image_to_camera(uv, depth, camera.intr)
    end_cam = image_to_camera(uv_end, depth, camera.intr)
    displacement = np.linalg.norm(end_cam - start_cam)
    if displacement <= 0:
        return 0.0
    return float(displacement / exposure_sec)


def map_candidate_to_track(
    candidate: FmoCandidate,
    roi: Tuple[int, int, int, int],
    camera: CameraModel,
    ball_diameter_m: float,
    fps: float,
    conf: float,
    prev_track: Optional[FmoTrack] = None,
    exposure_sec: Optional[float] = None,
) -> FmoTrack:
    """Convert an FMO detection into a 3D track estimate."""

    x, y, _, _ = roi
    uv_image = (candidate.center_uv[0] + x, candidate.center_uv[1] + y)
    diam_px = max(candidate.width_px, 2.0 * candidate.dt_peak)
    depth = depth_from_diameter(diam_px, ball_diameter_m, camera.intr)
    point_cam = image_to_camera(uv_image, depth, camera.intr)
    point_world = camera_to_world(point_cam, camera.extr)

    velocity = 0.0
    if prev_track is not None and fps > 0:
        dt = 1.0 / fps
        displacement = np.linalg.norm(np.asarray(point_world) - np.asarray(prev_track.xyz))
        if dt > 0:
            velocity = float(displacement / dt)

    if exposure_sec is not None:
        v_exp = _exposure_velocity(uv_image, candidate.axis_dir, candidate.length_px, depth, camera, exposure_sec)
        if velocity > 0:
            velocity = (velocity + v_exp) * 0.5
        else:
            velocity = v_exp

    return FmoTrack(
        uv=uv_image,
        xyz=tuple(map(float, point_world)),
        width_px=float(candidate.width_px),
        length_px=float(candidate.length_px),
        z_from_diam=float(depth),
        v_mps=float(velocity),
        conf=float(conf),
        axis_dir_2d=tuple(map(float, _axis_unit(candidate.axis_dir))),
        bbox_roi=candidate.bbox_roi,
    )


__all__ = ["FmoTrack", "map_candidate_to_track"]

"""Strike-zone decision helpers.

This module provides an adapter around the legacy "two plane intersection"
strike detection logic found in the historical notebooks of the project.  The
adapter exposes a compact state machine that tracks whether the ball has
successfully traversed the two decision planes and whether its projection lies
inside the corresponding polygons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


Vector3 = np.ndarray
def _normalise(v: Vector3) -> Vector3:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _point_in_polygon_2d(point: np.ndarray, polygon: Sequence[np.ndarray]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if cond:
            inside = not inside
    return inside


@dataclass
class PlanePolygon:
    """Helper storing a strike decision plane and its silhouette."""

    normal: Vector3
    offset: float
    polygon_world: np.ndarray
    origin: Vector3
    basis_u: Vector3
    basis_v: Vector3
    polygon_2d: np.ndarray

    @classmethod
    def from_points(
        cls, plane_points: Sequence[Vector3], polygon_points: Sequence[Vector3]
    ) -> "PlanePolygon":
        if len(plane_points) < 3:
            raise ValueError("At least three points required to define a plane")
        p0, p1, p2 = plane_points[:3]
        normal = _normalise(np.cross(p1 - p0, p2 - p0))
        offset = -float(np.dot(normal, p0))

        origin = polygon_points[0]
        u = _normalise(polygon_points[1] - origin)
        v = _normalise(np.cross(normal, u))
        if np.linalg.norm(v) == 0:
            # The polygon might be degenerate; fallback to another axis.
            arbitrary = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(arbitrary, normal)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0], dtype=float)
            v = _normalise(np.cross(normal, arbitrary))
            u = _normalise(np.cross(v, normal))

        poly_2d = []
        for pt in polygon_points:
            rel = pt - origin
            poly_2d.append(np.array([np.dot(rel, u), np.dot(rel, v)], dtype=float))

        return cls(
            normal=normal,
            offset=offset,
            polygon_world=np.asarray(polygon_points, dtype=float),
            origin=origin,
            basis_u=u,
            basis_v=v,
            polygon_2d=np.asarray(poly_2d, dtype=float),
        )

    def signed_distance(self, point: Vector3) -> float:
        return float(np.dot(self.normal, point) + self.offset)

    def project_to_plane(self, point: Vector3) -> np.ndarray:
        distance = self.signed_distance(point)
        on_plane = point - distance * self.normal
        rel = on_plane - self.origin
        return np.array([np.dot(rel, self.basis_u), np.dot(rel, self.basis_v)], dtype=float)

    def contains(self, point: Vector3, tol: float = 1e-4) -> bool:
        distance = abs(self.signed_distance(point))
        if distance > tol:
            return False
        projected = self.project_to_plane(point)
        return _point_in_polygon_2d(projected, self.polygon_2d)


@dataclass
class StrikeState:
    """State descriptor returned by :func:`StrikeZoneAdapter.update_and_check`."""

    passed_plane1: bool = False
    passed_plane2: bool = False
    inside_poly1: bool = False
    inside_poly2: bool = False
    is_strike: bool = False
    last_point: Optional[Vector3] = None
    last_distance_plane1: Optional[float] = None
    last_distance_plane2: Optional[float] = None


class StrikeZoneAdapter:
    """Wrap the legacy plane-polygon strike check into a reusable object."""

    def __init__(
        self,
        plane1_points: Sequence[Vector3],
        plane2_points: Sequence[Vector3],
        polygon1: Sequence[Vector3],
        polygon2: Sequence[Vector3],
        plane_tol: float = 1e-3,
    ) -> None:
        self.plane1 = PlanePolygon.from_points(plane1_points, polygon1)
        self.plane2 = PlanePolygon.from_points(plane2_points, polygon2)
        self.plane_tol = plane_tol

    def update_and_check(
        self, point_world: Vector3, prev_state: Optional[StrikeState] = None
    ) -> StrikeState:
        prev_state = prev_state or StrikeState()

        dist1 = self.plane1.signed_distance(point_world)
        dist2 = self.plane2.signed_distance(point_world)

        inside1 = self.plane1.contains(point_world, tol=self.plane_tol)
        inside2 = self.plane2.contains(point_world, tol=self.plane_tol)

        passed1 = prev_state.passed_plane1 or self._has_crossed(
            prev_state.last_distance_plane1, dist1
        )
        passed2 = prev_state.passed_plane2 or self._has_crossed(
            prev_state.last_distance_plane2, dist2
        )

        is_strike = passed1 and passed2 and inside1 and inside2

        return StrikeState(
            passed_plane1=passed1,
            passed_plane2=passed2,
            inside_poly1=inside1,
            inside_poly2=inside2,
            is_strike=is_strike,
            last_point=np.asarray(point_world, dtype=float),
            last_distance_plane1=dist1,
            last_distance_plane2=dist2,
        )

    @staticmethod
    def _has_crossed(previous: Optional[float], current: float) -> bool:
        if previous is None:
            return False
        return (previous > 0 and current <= 0) or (previous < 0 and current >= 0)


__all__ = ["StrikeZoneAdapter", "StrikeState"]

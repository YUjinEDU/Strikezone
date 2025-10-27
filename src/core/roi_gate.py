"""Region-of-interest gating based on world-space strike zone geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from geometry.camera import CameraModel
from utils.logging import get_logger


@dataclass
class ROIGate:
    """Compute image-plane regions of interest from strike zone planes."""

    camera: CameraModel
    margin_px: int = 40
    mode: str = "global"

    def __post_init__(self) -> None:
        self._logger = get_logger("roi_gate")
        self._last_roi: Optional[Tuple[int, int, int, int]] = None

    def compute(
        self,
        pitcher_plane: Sequence[Sequence[float]],
        catcher_plane: Sequence[Sequence[float]],
        frame_shape: Tuple[int, int, int],
        debug_frame: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int, int]:
        """Compute the ROI bounding box.

        Args:
            pitcher_plane: Quadrilateral on the pitcher-side strike zone plane.
            catcher_plane: Quadrilateral on the catcher-side strike zone plane.
            frame_shape: Shape tuple of the frame (rows, cols, channels).
            debug_frame: Optional frame copy used to draw debug overlays.

        Returns:
            ROI bounding box as (x, y, width, height).
        """

        height, width = frame_shape[:2]
        points_world = np.vstack([pitcher_plane, catcher_plane])
        img_points = self.camera.project(points_world)

        x_min = np.clip(np.min(img_points[:, 0]), 0, width - 1)
        x_max = np.clip(np.max(img_points[:, 0]), 0, width - 1)
        y_min = np.clip(np.min(img_points[:, 1]), 0, height - 1)
        y_max = np.clip(np.max(img_points[:, 1]), 0, height - 1)

        roi = self._expand_roi((x_min, y_min, x_max, y_max), width, height)
        self._last_roi = roi

        if debug_frame is not None:
            self._draw_debug(debug_frame, roi)

        return roi

    def _expand_roi(
        self, bounds: Tuple[float, float, float, float], frame_width: int, frame_height: int
    ) -> Tuple[int, int, int, int]:
        margin = float(self.margin_px)
        if self.mode == "expanded":
            margin *= 1.5
        elif self.mode == "debug":
            margin *= 2.0

        x_min, y_min, x_max, y_max = bounds
        x_min = max(0, int(np.floor(x_min - margin)))
        y_min = max(0, int(np.floor(y_min - margin)))
        x_max = min(frame_width - 1, int(np.ceil(x_max + margin)))
        y_max = min(frame_height - 1, int(np.ceil(y_max + margin)))

        width = max(1, x_max - x_min)
        height = max(1, y_max - y_min)
        return x_min, y_min, width, height

    def _draw_debug(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ROI {self.mode}",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


__all__ = ["ROIGate"]

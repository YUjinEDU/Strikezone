"""Visualization helpers for the strike-zone pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - OpenCV might be unavailable in minimal environments
    import cv2
except Exception:  # pragma: no cover - we gracefully degrade without OpenCV
    cv2 = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class PrincipalAxis:
    """Description of a principal component in image space."""

    center: Tuple[float, float]
    direction: Tuple[float, float]
    length: float


@dataclass
class OverlayData:
    """Information rendered by :class:`FrameDebugVisualizer`."""

    roi: Optional[Tuple[int, int, int, int]] = None
    dt_peak: Optional[Tuple[int, int]] = None
    pca_axis: Optional[PrincipalAxis] = None
    xyz: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    plane_silhouettes: Sequence[Sequence[Tuple[float, float]]] = field(
        default_factory=list
    )
    diff: Optional[np.ndarray] = None
    dt_map: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None


class FrameDebugVisualizer:
    """Interactive debug window with stage toggles and optional video capture."""

    STAGES = ("original", "diff", "dt", "binary", "overlay")

    def __init__(
        self,
        window_name: str = "Strikezone Debug",
        save_path: Optional[str] = None,
        save_fps: float = 60.0,
    ) -> None:
        self.window_name = window_name
        self.save_path = save_path
        self.save_fps = save_fps
        self._stage_index = len(self.STAGES) - 1  # default to overlay
        self._writer: Optional["cv2.VideoWriter"] = None
        self._last_key: Optional[int] = None
        if cv2 is None:
            LOGGER.warning(
                "OpenCV is not available – FrameDebugVisualizer cannot display windows"
            )
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def close(self) -> None:
        if cv2 is None:
            return
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        cv2.destroyWindow(self.window_name)

    def render(self, frame: np.ndarray, overlay: OverlayData) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for visualization")

        stage_frames = self._compose_stage_frames(frame, overlay)
        display = stage_frames[self._stage_index]

        cv2.imshow(self.window_name, display)
        self._ensure_writer(stage_frames[-1])
        if self._writer is not None:
            self._writer.write(stage_frames[-1])

        return display

    def handle_key(self, key: int, overlay: OverlayData) -> None:
        if cv2 is None:
            return

        if key == ord("d"):
            self._stage_index = (self._stage_index + 1) % len(self.STAGES)
        elif key == ord("b") and self._last_key != ord("b"):
            if overlay.xyz is not None:
                print(
                    "Ball XYZ:",
                    np.array2string(np.asarray(overlay.xyz), precision=3),
                )
        if key == -1:
            self._last_key = None
        else:
            self._last_key = key

    # ------------------------------------------------------------------
    def _compose_stage_frames(
        self, frame: np.ndarray, overlay: OverlayData
    ) -> List[np.ndarray]:
        color_frame = self._ensure_color(frame)
        diff_frame = self._prepare_optional_frame(overlay.diff, color_frame)
        dt_frame = self._prepare_dt_frame(overlay.dt_map, color_frame)
        binary_frame = self._prepare_binary_frame(overlay.binary, color_frame)
        overlay_frame = self._draw_overlay(color_frame.copy(), overlay)
        return [color_frame, diff_frame, dt_frame, binary_frame, overlay_frame]

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self.save_path is None or cv2 is None:
            return
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        for fourcc in ("avc1", "mp4v", "H264", "X264"):
            writer = cv2.VideoWriter(
                self.save_path,
                cv2.VideoWriter_fourcc(*fourcc),
                self.save_fps,
                (w, h),
            )
            if writer.isOpened():
                self._writer = writer
                break
            writer.release()
        if self._writer is None:
            LOGGER.error("Failed to open video writer for %s", self.save_path)

    @staticmethod
    def _ensure_color(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.dstack([
                frame,
                frame,
                frame,
            ])
        return frame.copy()

    @staticmethod
    def _prepare_optional_frame(
        optional: Optional[np.ndarray], fallback: np.ndarray
    ) -> np.ndarray:
        if optional is None:
            return fallback.copy()
        frame = optional
        if frame.ndim == 2:
            if cv2 is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.dstack([frame, frame, frame])
        return frame.copy()

    @staticmethod
    def _prepare_dt_frame(
        dt_map: Optional[np.ndarray], fallback: np.ndarray
    ) -> np.ndarray:
        if dt_map is None:
            return fallback.copy()
        dt = dt_map.astype(np.float32)
        dt = dt - dt.min()
        if dt.max() > 0:
            dt /= dt.max()
        dt_uint8 = np.clip(dt * 255.0, 0, 255).astype(np.uint8)
        if cv2 is not None:
            colored = cv2.applyColorMap(dt_uint8, cv2.COLORMAP_JET)
        else:
            colored = np.dstack([dt_uint8] * 3)
        return colored

    @staticmethod
    def _prepare_binary_frame(
        binary: Optional[np.ndarray], fallback: np.ndarray
    ) -> np.ndarray:
        if binary is None:
            return fallback.copy()
        mask = np.clip(binary.astype(np.uint8) * 255, 0, 255)
        if cv2 is not None:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.dstack([mask] * 3)

    def _draw_overlay(self, frame: np.ndarray, overlay: OverlayData) -> np.ndarray:
        if cv2 is None:
            return frame
        if overlay.roi is not None:
            x, y, w, h = overlay.roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if overlay.dt_peak is not None:
            cv2.circle(frame, tuple(map(int, overlay.dt_peak)), 4, (0, 0, 255), -1)
        if overlay.pca_axis is not None:
            axis = overlay.pca_axis
            dir_vec = np.asarray(axis.direction, dtype=float)
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
            center = np.asarray(axis.center, dtype=float)
            half = dir_vec * (axis.length / 2.0)
            pt1 = tuple(np.round(center - half).astype(int))
            pt2 = tuple(np.round(center + half).astype(int))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA)
        if overlay.plane_silhouettes:
            for poly in overlay.plane_silhouettes:
                pts = np.asarray(poly, dtype=float)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                pts_int = np.round(pts).astype(int)
                cv2.polylines(frame, [pts_int], isClosed=True, color=(255, 255, 0), thickness=1)
        self._draw_text(frame, overlay)
        return frame

    def _draw_text(self, frame: np.ndarray, overlay: OverlayData) -> None:
        if cv2 is None:
            return
        lines: List[str] = []
        if overlay.frame_index is not None:
            lines.append(f"frame: {overlay.frame_index}")
        if overlay.timestamp is not None:
            lines.append(f"t: {overlay.timestamp:.3f}s")
        if overlay.xyz is not None:
            xyz = np.asarray(overlay.xyz)
            lines.append(
                "xyz: "
                + np.array2string(xyz, formatter={"float_kind": lambda v: f"{v:6.3f}"})
            )
        if overlay.velocity is not None:
            vel = np.asarray(overlay.velocity)
            speed = np.linalg.norm(vel)
            lines.append(
                "v: "
                + np.array2string(
                    vel,
                    formatter={"float_kind": lambda v: f"{v:6.2f}"},
                )
                + f" |{speed:5.2f} m/s"
            )
        if overlay.confidence is not None:
            lines.append(f"conf: {overlay.confidence:.2f}")

        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, 30 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


__all__ = ["FrameDebugVisualizer", "OverlayData", "PrincipalAxis"]

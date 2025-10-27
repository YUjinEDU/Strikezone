"""Fast moving object detection pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import ndimage as ndi

from .config import CameraIntrinsics, PipelineConfig
from .depth import back_project_uvz_to_xyz
from .distance import estimate_radius_from_dt


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Candidate:
    """Detected fast moving object candidate."""

    frame_index: int
    center_uv: np.ndarray
    confidence: float
    radius_px: float
    depth_m: float
    velocity_mps: np.ndarray


class StrikeZonePipeline:
    """Implements the StrikeZone fast moving object pipeline."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: PipelineConfig,
    ) -> None:
        self.intrinsics = intrinsics
        self.config = config
        self._previous_gray: np.ndarray | None = None
        self._previous_position: np.ndarray | None = None

    def _normalize_frames(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        return frame

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114], dtype=frame.dtype))

    def _prepare_mask(self, gray: np.ndarray) -> np.ndarray:
        contrast = 1.0 - gray
        mask = contrast > self.config.dfactor
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_fill_holes(mask)
        return mask

    def _compute_confidence(
        self,
        mask: np.ndarray,
        gray: np.ndarray,
        radius_px: float,
        motion_strength: float,
    ) -> float:
        contrast = 1.0 - gray
        if np.any(mask):
            color_score = float(np.clip(np.mean(contrast[mask]) * 2.0, 0.0, 1.0))
        else:
            color_score = 0.0
        shape_score = float(np.clip((2.0 * radius_px) / self.config.max_width_px, 0.0, 1.0))
        raw = (
            self.config.alpha * motion_strength
            + self.config.beta * color_score
            + self.config.gamma * shape_score
        )
        return float(_sigmoid(raw))

    def detect(self, frames: Sequence[np.ndarray]) -> List[Candidate]:
        candidates: List[Candidate] = []
        self._previous_gray = None
        self._previous_position = None

        for index, frame in enumerate(frames):
            norm = self._normalize_frames(frame)
            gray = self._to_grayscale(norm)

            motion_strength = 0.0
            if self._previous_gray is not None:
                diff = np.abs(gray - self._previous_gray)
                motion_strength = float(np.clip(diff.mean() * 8.0, 0.0, 1.0))
            self._previous_gray = gray

            mask = self._prepare_mask(gray)
            if np.count_nonzero(mask) < self.config.min_len_px:
                continue

            labeled, num = ndi.label(mask)
            if num == 0:
                continue

            for label in range(1, num + 1):
                component = labeled == label
                if np.count_nonzero(component) < self.config.min_len_px:
                    continue

                radius_px = estimate_radius_from_dt(component)
                if radius_px == 0:
                    continue

                diameter_px = 2.0 * radius_px
                if diameter_px > self.config.max_width_px:
                    continue

                coords = np.argwhere(component)
                center_yx = coords.mean(axis=0)
                center_uv = np.array([center_yx[1], center_yx[0]], dtype=float)

                depth_m = (self.intrinsics.fx * self.config.ball_diameter_m) / max(diameter_px, 1e-6)
                position = back_project_uvz_to_xyz(center_uv[0], center_uv[1], depth_m, self.intrinsics)

                velocity = np.zeros(3, dtype=float)
                if self._previous_position is not None:
                    velocity = (position - self._previous_position) * self.config.frame_rate
                self._previous_position = position

                confidence = self._compute_confidence(component, gray, radius_px, motion_strength)
                if confidence < 0.5:
                    continue

                candidates.append(
                    Candidate(
                        frame_index=index,
                        center_uv=center_uv,
                        confidence=confidence,
                        radius_px=radius_px,
                        depth_m=depth_m,
                        velocity_mps=velocity,
                    )
                )

        return candidates

"""Visualization helpers for strikezone overlays."""
from __future__ import annotations

import cv2
import numpy as np

from typing import Dict, Tuple


Color = Tuple[int, int, int]

DEFAULT_COLORS: Dict[str, Color] = {
    "roi": (0, 255, 0),
    "candidate": (0, 165, 255),
    "strike": (0, 0, 255),
    "ball": (255, 255, 0),
}


def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], color: Color = DEFAULT_COLORS["roi"]) -> None:
    """Draw an ROI rectangle on the frame in-place."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], color: Color = (255, 255, 255)) -> None:
    """Draw a text label onto the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: Color = (0, 0, 255), alpha: float = 0.4) -> None:
    """Blend a binary mask onto the frame in-place."""
    colored = np.zeros_like(frame)
    colored[:, :] = color
    mask_bool = mask.astype(bool)
    frame[mask_bool] = cv2.addWeighted(frame[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0)


def draw_velocity_vector(frame: np.ndarray, origin: Tuple[int, int], velocity: Tuple[float, float], scale: float = 0.1, color: Color = (255, 0, 0)) -> None:
    """Render a velocity arrow representing pixel velocity."""
    end_point = (int(origin[0] + velocity[0] * scale), int(origin[1] + velocity[1] * scale))
    cv2.arrowedLine(frame, origin, end_point, color, 2, tipLength=0.3)

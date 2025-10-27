"""Application configuration management utilities.

This module defines the :class:`AppConfig` data model, which centralises all
runtime parameters for the fast-moving object (FMO) detection pipeline. The
configuration is implemented with `pydantic` in order to provide validation and
rich default values for downstream components. Two helper functions are
exposed: :func:`load_config` to load a YAML configuration file and
:func:`default_config` to obtain a ready-to-use configuration instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, Field, validator


class AppConfig(BaseModel):
    """Pydantic model describing the application configuration.

    Attributes:
        source: Video input identifier. Integer values select a camera index
            while strings are interpreted as video file paths.
        width: Expected input frame width in pixels.
        height: Expected input frame height in pixels.
        fps: Target frame rate in frames per second.
        exposure_sec: Exposure time in seconds.
        use_gpu: Flag indicating whether GPU acceleration should be attempted.
        fx: Focal length in pixels along the x axis.
        fy: Focal length in pixels along the y axis.
        cx: Principal point x coordinate in pixels.
        cy: Principal point y coordinate in pixels.
        dist_coeffs: Radial/tangential distortion coefficients (k1, k2, p1, p2, k3).
        ball_diameter_m: Regulation baseball diameter in metres.
        roi_mode: ROI generation mode.
        roi_margin_px: Padding added around the ROI bounding box in pixels.
        dfactor: Down-scaling factor for the first-pass FMO detector.
        min_len_px: Minimum streak length considered a valid detection in pixels.
        max_width_px: Maximum allowed streak width in pixels.
        min_speed_px_per_frame: Minimum apparent speed threshold in pixels/frame.
        bg_window: Window size for the rolling median background in frames.
        bg_freeze_frames: Number of frames to freeze background updates after
            motion is detected.
        hsv_priors: Sequence of HSV ranges used as colour priors for the ball.
        alpha: Logistic fusion parameter controlling colour prior strength.
        beta: Logistic fusion parameter controlling FMO detector contribution.
        gamma: Logistic fusion bias term.
        enable_tbd: Enable Trajectory by Detection (TbD) refinement if available.
        refine_near_planes: Whether TbD refinement should consider planes near the
            strike zone boundaries.
        refine_window: Number of frames to aggregate when refining detections.
    """

    source: Union[int, str] = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    exposure_sec: float = Field(0.0015, gt=0.0)
    use_gpu: bool = False

    fx: float = 930.0
    fy: float = 930.0
    cx: float = 320.0
    cy: float = 240.0
    dist_coeffs: Tuple[float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    ball_diameter_m: float = Field(0.073, gt=0.0)

    roi_mode: str = Field("global", regex=r"^(global|expanded|debug)$")
    roi_margin_px: int = Field(40, ge=0)

    dfactor: float = Field(0.45, gt=0.0, lt=1.0)
    min_len_px: int = Field(12, ge=1)
    max_width_px: int = Field(30, ge=1)
    min_speed_px_per_frame: float = Field(8.0, ge=0.0)

    bg_window: int = Field(5, ge=1)
    bg_freeze_frames: int = Field(3, ge=0)

    hsv_priors: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = Field(
        default_factory=lambda: [
            ((0, 0, 160), (179, 40, 255)),
        ]
    )

    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = -1.0

    enable_tbd: bool = True
    refine_near_planes: bool = True
    refine_window: int = Field(2, ge=1)

    @validator("hsv_priors")
    def _validate_priors(
        cls, value: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    ) -> Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        for low, high in value:
            if len(low) != 3 or len(high) != 3:
                raise ValueError("HSV bounds must be triplets")
            if not all(0 <= v <= 255 for v in (*low, *high)):
                raise ValueError("HSV bounds must be in [0, 255]")
        return value

    class Config:
        """Pydantic configuration options."""

        arbitrary_types_allowed = True
        validate_assignment = True


def load_config(path: Union[str, Path]) -> AppConfig:
    """Load an :class:`AppConfig` instance from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Loaded :class:`AppConfig` instance. If the file cannot be read or parsed
        the exception is propagated to the caller.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return AppConfig.parse_obj(data)


def default_config() -> AppConfig:
    """Return a default :class:`AppConfig` instance."""

    return AppConfig()


__all__ = ["AppConfig", "load_config", "default_config"]

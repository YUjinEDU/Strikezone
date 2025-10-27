"""Configuration schema and loader for the Strikezone pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, validator


class CameraConfig(BaseModel):
    """Camera configuration parameters."""

    source: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    exposure_sec: Optional[float] = Field(default=None, ge=0)


class ProcessingConfig(BaseModel):
    """Processing parameters for ROI and detection."""

    roi_mode: str = Field("expanded", regex="^(expanded|tight|manual)$")
    roi_expand: float = Field(0.2, ge=0, le=1)
    dfactor: float = Field(0.45, gt=0, lt=1)
    enable_tbd: bool = False
    buffer_history: int = Field(5, ge=1)


class OutputConfig(BaseModel):
    """Output related configuration."""

    save_path: Optional[Path] = None
    enable_debug: bool = False


class Settings(BaseModel):
    """Application level settings aggregated from YAML and CLI."""

    camera: CameraConfig = CameraConfig()
    processing: ProcessingConfig = ProcessingConfig()
    output: OutputConfig = OutputConfig()

    @validator("camera")
    def _validate_camera(cls, value: CameraConfig) -> CameraConfig:
        if value.width <= 0 or value.height <= 0:
            raise ValueError("Camera resolution must be positive")
        if value.fps <= 0:
            raise ValueError("FPS must be positive")
        return value


def load_settings(path: Optional[Path]) -> Settings:
    """Load a configuration file into :class:`Settings`.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file. When ``None`` defaults are
        returned.

    Returns
    -------
    Settings
        Parsed configuration.
    """

    if path is None:
        return Settings()

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Settings.parse_obj(data)

"""Configuration helpers for the StrikeZone pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    """Simple container for pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable parameters for the pipeline."""

    dfactor: float = 0.2
    min_len_px: int = 12
    max_width_px: int = 80
    alpha: float = 3.0
    beta: float = 2.0
    gamma: float = 1.5
    frame_rate: float = 60.0
    ball_diameter_m: float = 0.074

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create a config from a nested dictionary."""

        return cls(**data)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return PipelineConfig.from_dict(data)

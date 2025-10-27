"""Strike zone adapter integrating with external logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class StrikeDecision:
    """Decision output of the strike zone evaluator."""

    is_strike: bool
    confidence: float


class StrikeZoneEvaluator:
    """Evaluate whether a ball trajectory intersects the strike zone."""

    def __init__(self, zone_min: Tuple[float, float, float], zone_max: Tuple[float, float, float]) -> None:
        self.zone_min = np.array(zone_min, dtype=np.float32)
        self.zone_max = np.array(zone_max, dtype=np.float32)

    def evaluate(self, position: np.ndarray) -> StrikeDecision:
        """Return strike decision based on position inside zone bounds."""
        inside = np.all(position >= self.zone_min) and np.all(position <= self.zone_max)
        confidence = 0.9 if inside else 0.1
        return StrikeDecision(is_strike=bool(inside), confidence=confidence)

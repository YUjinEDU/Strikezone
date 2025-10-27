"""Optional TbD based refinement with graceful fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:  # Optional torch dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class TbDConfig:
    """Configuration for the TbD refiner."""

    iterations: int = 3
    sigma: float = 1.0


class TbDRefiner:
    """Apply TbD style refinement when the optional dependency is present."""

    def __init__(self, config: TbDConfig) -> None:
        self.config = config
        self.enabled = torch is not None

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """Refine a candidate mask using iterative smoothing."""
        if mask.size == 0:
            return mask
        if not self.enabled:
            return cv2.medianBlur(mask, 3)

        tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        kernel_size = int(2 * round(3 * self.config.sigma) + 1)
        kernel = self._gaussian_kernel(kernel_size, self.config.sigma)
        kernel = kernel.to(tensor.device)
        result = tensor
        for _ in range(self.config.iterations):
            result = torch.nn.functional.conv2d(result, kernel, padding=kernel_size // 2)
        refined = (result.squeeze().numpy() > 0.3).astype(np.uint8) * 255
        return refined

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> "torch.Tensor":  # type: ignore[name-defined]
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

"""TbD-based refinement utilities.

This module wraps the optional :mod:`deblatting_python` implementation of
Track-before-detect (TbD) motion deblatting.  The intent is to provide a thin
adaptor that can consume the per-frame candidate stack produced by the coarse
tracker and to return sub-frame accurate position and velocity updates.

The actual TbD implementation is fairly heavyweight and only available from the
`deblatting_python` repository.  Network access is not guaranteed in the runtime
environment, therefore all imports are best-effort and the refiner silently
falls back to a no-op update whenever the dependency is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from deblatting_python import run as _tbd_run  # type: ignore
except Exception:  # pragma: no cover - it is fine if the dependency is missing
    _tbd_run = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@dataclass
class FrameObservation:
    """Container describing an observation around the strike decision moment.

    Parameters
    ----------
    frame_index:
        Global frame index inside the processed sequence.
    image:
        Full frame (BGR) image as a ``H×W×3`` numpy array.
    mask:
        Binary ROI mask with the same spatial dimensions as ``image``.  Pixels
        belonging to the candidate ball should contain ``1`` and all other
        pixels ``0``.
    roi:
        Region-of-interest bounding box ``(x, y, width, height)`` measured in
        pixels.  All TbD processing runs inside this ROI.
    timestamp:
        Timestamp in seconds.  The time base is arbitrary but must be
        monotonically increasing.
    between_planes:
        Flag indicating whether the observation lies between the strike decision
        planes.  Only such frames will be considered for refinement.
    pixel_to_meter:
        Optional scaling factor converting the ROI's diagonal in pixels to
        metres.  The refiner uses it to map the TbD kernel blur width to metric
        velocity estimates.
    """

    frame_index: int
    image: np.ndarray
    mask: np.ndarray
    roi: Tuple[int, int, int, int]
    timestamp: float
    between_planes: bool = False
    pixel_to_meter: Optional[float] = None

    def crop(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return cropped image and mask restricted to the ROI."""

        x, y, w, h = self.roi
        cropped_img = self.image[y : y + h, x : x + w]
        cropped_mask = self.mask[y : y + h, x : x + w]
        return cropped_img, cropped_mask


@dataclass
class TrackEstimate:
    """Minimal kinematic description of the tracked baseball."""

    xyz: np.ndarray
    velocity_mps: np.ndarray
    confidence: float
    timestamp: float


@dataclass
class TbDResult:
    """Return container for :class:`TbdRefiner.refine` outcomes."""

    estimate: TrackEstimate
    stack_indices: Sequence[int]
    kernel: Optional[np.ndarray]
    success: bool
    message: Optional[str] = None
    raw_output: Optional[object] = None


@dataclass
class TbdRefinerConfig:
    """Configuration options for :class:`TbdRefiner`."""

    refine_window: int = 2
    median_window: int = 5
    min_stack: int = 3
    eps: float = 1e-6
    max_kernel_axis_ratio: float = 20.0
    fallback_confidence: float = 0.25


class TbDUnavailable(RuntimeError):
    """Raised when TbD execution is requested but the dependency is missing."""


class TbdRefiner:
    """Refine coarse baseball tracks using motion deblatting.

    The refiner collects a temporal stack around the moment the ball travels
    between the two strike decision planes, estimates the local background via a
    small temporal median, and feeds the observation to TbD.  The recovered
    motion kernel is analysed to adjust the sub-frame position and velocity.

    Parameters
    ----------
    config:
        Configuration controlling the stack length and numerical safeguards.
    runner:
        Optional callable to execute the TbD algorithm.  When ``None`` the
        refiner tries to import :mod:`deblatting_python.run`.
    """

    def __init__(
        self,
        config: Optional[TbdRefinerConfig] = None,
        runner: Optional[Callable[..., object]] = None,
    ) -> None:
        self.config = config or TbdRefinerConfig()
        self._runner = runner or _tbd_run
        self._available = self._runner is not None
        if not self._available:
            LOGGER.warning(
                "deblatting_python is not available – TbD refinement disabled"
            )

    @property
    def available(self) -> bool:
        """Return ``True`` when a TbD backend is ready for use."""

        return self._available

    def refine(
        self,
        observations: Sequence[FrameObservation],
        coarse_estimate: TrackEstimate,
    ) -> TbDResult:
        """Run TbD refinement on the supplied observation stack.

        Parameters
        ----------
        observations:
            Sequence of frame observations ordered in time.
        coarse_estimate:
            Track estimate produced by the coarse tracker.

        Returns
        -------
        TbDResult
            Result structure containing the refined estimate.  When TbD is not
            available the original estimate is returned with ``success`` set to
            ``False`` and an explanatory message.
        """

        if not self.available:
            return TbDResult(
                estimate=coarse_estimate,
                stack_indices=(),
                kernel=None,
                success=False,
                message="TbD backend unavailable",
            )

        stack = self._build_stack(observations)
        if not stack:
            return TbDResult(
                estimate=coarse_estimate,
                stack_indices=(),
                kernel=None,
                success=False,
                message="insufficient frames near decision planes",
            )

        images: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        stack_indices: List[int] = []
        for obs in stack:
            cropped_img, cropped_mask = obs.crop()
            images.append(cropped_img.astype(np.float32) / 255.0)
            masks.append(cropped_mask.astype(np.float32))
            stack_indices.append(obs.frame_index)

        background = self._estimate_background(images)
        observation = np.stack(images, axis=0)
        mask = np.clip(np.stack(masks, axis=0), 0.0, 1.0)

        try:
            raw_output = self._execute_tbd(observation, background, mask)
        except TbDUnavailable as exc:
            return TbDResult(
                estimate=coarse_estimate,
                stack_indices=stack_indices,
                kernel=None,
                success=False,
                message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("TbD execution failed: %%s", exc)
            return TbDResult(
                estimate=coarse_estimate,
                stack_indices=stack_indices,
                kernel=None,
                success=False,
                message=f"TbD execution failed: {exc}",
            )

        kernel = self._extract_kernel(raw_output)
        if kernel is None:
            return TbDResult(
                estimate=coarse_estimate,
                stack_indices=stack_indices,
                kernel=None,
                success=False,
                message="TbD output did not contain a motion kernel",
                raw_output=raw_output,
            )

        axis_center, axis_width = self._analyse_kernel(kernel)
        refined = self._apply_adjustment(stack, coarse_estimate, axis_center, axis_width)

        return TbDResult(
            estimate=refined,
            stack_indices=stack_indices,
            kernel=kernel,
            success=True,
            raw_output=raw_output,
        )

    # ------------------------------------------------------------------
    # Stack construction helpers
    def _build_stack(
        self, observations: Sequence[FrameObservation]
    ) -> List[FrameObservation]:
        window = self.config.refine_window
        candidates = [i for i, obs in enumerate(observations) if obs.between_planes]
        if not candidates:
            return []

        center_idx = candidates[len(candidates) // 2]
        start = max(0, center_idx - window)
        end = min(len(observations), center_idx + window + 1)
        stack = list(observations[start:end])
        if len(stack) < self.config.min_stack:
            return []
        return stack

    def _estimate_background(self, images: Sequence[np.ndarray]) -> np.ndarray:
        window = min(len(images), self.config.median_window)
        if window == 0:
            raise TbDUnavailable("no frames available for background estimation")
        subset = np.stack(images[:window], axis=0)
        background = np.median(subset, axis=0)
        return background.astype(np.float32)

    # ------------------------------------------------------------------
    # TbD execution helpers
    def _execute_tbd(
        self,
        observation: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
    ) -> object:
        if self._runner is None:
            raise TbDUnavailable("TbD runner missing")

        runner = self._runner
        kwargs = {
            "image_stack": observation,
            "background": background,
            "mask": mask,
            "device": "cpu",
        }

        try:
            return runner(**kwargs)
        except TypeError:
            # Try a slightly different, more generic signature often used by
            # research code (stack first, everything else packed into kwargs).
            return runner(observation, background=background, mask=mask, device="cpu")

    def _extract_kernel(self, raw_output: object) -> Optional[np.ndarray]:
        """Best-effort extraction of the motion kernel from TbD output."""

        if raw_output is None:
            return None

        if isinstance(raw_output, np.ndarray):
            return raw_output.astype(np.float32)

        if isinstance(raw_output, (list, tuple)) and raw_output:
            first = raw_output[0]
            if isinstance(first, np.ndarray):
                return first.astype(np.float32)

        if isinstance(raw_output, dict):
            for key in ("kernel", "motion_kernel", "H", "psf"):
                value = raw_output.get(key)
                if isinstance(value, np.ndarray):
                    return value.astype(np.float32)

        return None

    def _analyse_kernel(self, kernel: np.ndarray) -> Tuple[float, float]:
        """Return (sub-frame center offset, blur width) along the major axis."""

        if kernel.ndim != 2:
            kernel = kernel.squeeze()
        if kernel.ndim != 2:
            raise ValueError("TbD motion kernel must be 2-D")

        kernel = np.maximum(kernel, 0.0)
        total = float(kernel.sum())
        if total <= self.config.eps:
            return 0.0, 0.0

        kernel = kernel / total
        coords_y, coords_x = np.indices(kernel.shape, dtype=np.float32)
        center_x = float(np.sum(coords_x * kernel))
        center_y = float(np.sum(coords_y * kernel))

        samples = np.stack([coords_x - center_x, coords_y - center_y], axis=-1)
        cov = (samples * kernel[..., None]).reshape(-1, 2).T @ samples.reshape(-1, 2)
        cov = cov / max(total, self.config.eps)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Project the sub-frame center onto the dominant axis.  The kernel is in
        # pixel units; we keep the offset in fractions of the kernel width so
        # that callers can interpret it relative to their own scale.
        dominant_axis = eigvecs[:, 0]
        axis_center = float(np.dot([center_x - kernel.shape[1] / 2.0, center_y - kernel.shape[0] / 2.0], dominant_axis))
        axis_width = float(np.sqrt(max(eigvals[0], 0.0)))

        if eigvals[1] > 0 and eigvals[0] / eigvals[1] > self.config.max_kernel_axis_ratio:
            axis_width = float(np.sqrt(eigvals[1]))

        return axis_center, axis_width

    # ------------------------------------------------------------------
    def _apply_adjustment(
        self,
        stack: Sequence[FrameObservation],
        coarse_estimate: TrackEstimate,
        axis_center: float,
        axis_width: float,
    ) -> TrackEstimate:
        timestamps = [obs.timestamp for obs in stack]
        dt = 0.0
        if len(timestamps) >= 2:
            dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)

        direction = coarse_estimate.velocity_mps.astype(np.float32)
        speed = float(np.linalg.norm(direction))
        if speed > self.config.eps:
            direction /= speed
        else:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        delta_t = axis_center * dt
        delta_pos = direction * speed * delta_t

        # Map the blur width (pixels) into a velocity magnitude adjustment.
        pixel_to_meter = self._estimate_pixel_to_meter(stack)
        if pixel_to_meter is not None and dt > self.config.eps:
            blur_meters = axis_width * pixel_to_meter
            refined_speed = blur_meters / dt
        else:
            refined_speed = speed

        if refined_speed <= self.config.eps:
            refined_speed = max(speed, self.config.fallback_confidence)

        refined_velocity = direction * refined_speed
        refined_position = coarse_estimate.xyz + delta_pos

        confidence = max(coarse_estimate.confidence, self.config.fallback_confidence)

        return TrackEstimate(
            xyz=refined_position,
            velocity_mps=refined_velocity,
            confidence=confidence,
            timestamp=coarse_estimate.timestamp,
        )

    def _estimate_pixel_to_meter(
        self, stack: Sequence[FrameObservation]
    ) -> Optional[float]:
        scales = [obs.pixel_to_meter for obs in stack if obs.pixel_to_meter]
        if not scales:
            return None
        return float(np.median(scales))


__all__ = [
    "FrameObservation",
    "TrackEstimate",
    "TbDResult",
    "TbdRefiner",
    "TbdRefinerConfig",
]

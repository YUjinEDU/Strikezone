"""Frame stabilisation utilities leveraging fiducial markers."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from utils.logging import get_logger


@dataclass
class StabilizerState:
    """Container holding the current stabilisation state."""

    reference_h: Optional[np.ndarray] = None
    history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    missed_frames: int = 0


class FrameStabilizer:
    """Stabilises frames based on ArUco marker detections."""

    def __init__(
        self,
        warmup_frames: int = 30,
        max_interp_frames: int = 10,
        dictionary: int = cv2.aruco.DICT_4X4_50,
    ) -> None:
        """Initialise the frame stabiliser.

        Args:
            warmup_frames: Number of frames used to establish the reference homography.
            max_interp_frames: Maximum number of frames to interpolate when markers are lost.
            dictionary: OpenCV ArUco dictionary identifier used for detection.
        """
        self._warmup_frames = warmup_frames
        self._max_interp_frames = max_interp_frames
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
        try:
            self._parameters = cv2.aruco.DetectorParameters()
        except AttributeError:  # OpenCV < 4.7 fallback
            self._parameters = cv2.aruco.DetectorParameters_create()
        self._state = StabilizerState()
        self._frame_count = 0
        self._logger = get_logger("stabilizer")
        self._ref_corners: Optional[dict[int, np.ndarray]] = None

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """Stabilise a frame using ArUco detections.

        Args:
            frame: Input BGR frame.

        Returns:
            Tuple containing the stabilised frame, a success flag, and the
            current homography estimate.
        """

        self._frame_count += 1
        detection = self._detect_markers(frame)
        h_curr: Optional[np.ndarray]

        if detection is not None:
            ids, corners = detection
            if self._ref_corners is None:
                self._ref_corners = {int(i): c[0].copy() for i, c in zip(ids.flatten(), corners)}
                self._logger.info("Reference marker layout initialised with %d markers", len(corners))

            h_curr = self._estimate_homography(ids, corners)
            if h_curr is not None:
                self._state.history.append(h_curr)
                self._state.missed_frames = 0
                if self._frame_count <= self._warmup_frames or self._state.reference_h is None:
                    self._state.reference_h = h_curr.copy()
            else:
                h_curr = self._handle_missing()
        else:
            h_curr = self._handle_missing()

        if h_curr is None or self._state.reference_h is None:
            self._logger.warning("Stabilisation unavailable; returning original frame")
            return frame, False, h_curr

        try:
            warp = self._state.reference_h @ np.linalg.inv(h_curr)
        except np.linalg.LinAlgError:
            self._logger.error("Singular homography encountered; returning original frame")
            return frame, False, h_curr
        stabilised = cv2.warpPerspective(frame, warp, (frame.shape[1], frame.shape[0]))
        return stabilised, True, h_curr

    def _detect_markers(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, list[np.ndarray]]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary, parameters=self._parameters)
        if ids is None or len(ids) == 0:
            return None
        return ids, corners

    def _estimate_homography(
        self, ids: np.ndarray, corners: list[np.ndarray]
    ) -> Optional[np.ndarray]:
        if self._ref_corners is None:
            return None

        src_pts = []
        dst_pts = []
        for marker_id, corner in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id not in self._ref_corners:
                continue
            src_pts.extend(self._ref_corners[marker_id])
            dst_pts.extend(corner[0])

        if len(src_pts) < 4:
            return None

        h_curr, mask = cv2.findHomography(np.array(dst_pts), np.array(src_pts), cv2.RANSAC)
        if h_curr is None or (mask is not None and mask.sum() < 4):
            return None

        h_curr /= h_curr[2, 2]
        return h_curr

    def _handle_missing(self) -> Optional[np.ndarray]:
        self._state.missed_frames += 1
        if self._state.missed_frames > self._max_interp_frames:
            self._logger.warning("Marker tracking lost for too long; reset required")
            return None
        if len(self._state.history) == 0:
            return None
        if len(self._state.history) == 1:
            return self._state.history[-1]

        alpha = min(1.0, self._state.missed_frames / float(self._max_interp_frames))
        prev = self._state.history[-2]
        last = self._state.history[-1]
        interp = (1.0 - alpha) * prev + alpha * last
        if interp[2, 2] != 0:
            interp /= interp[2, 2]
        return interp


__all__ = ["FrameStabilizer", "StabilizerState"]

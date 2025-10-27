"""Entry point for the Strikezone FMO light pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.settings import CameraConfig, Settings, load_settings
from src.core.background import BackgroundModel
from src.core.roi_gate import ROIGate, ROIState
from src.core.stabilizer import FrameStabilizer
from src.detection.fmo_light import FmoLightDetector
from src.fusion.color_prior import ColorPrior, ColorPriorConfig
from src.geometry.camera import CameraParameters
from src.geometry.mapping import estimate_trajectory
from src.refine.tbd_refiner import TbDConfig, TbDRefiner
from src.strike.zone import StrikeZoneEvaluator
from src.utils import vis
from src.utils.logging import configure_logging, get_logger


LOGGER = get_logger(__name__)


class ToggleState:
    """Maintain keyboard toggle states."""

    def __init__(self) -> None:
        self.debug = False
        self.color_preview = False
        self.background_preview = False
        self.paused = False


class PipelineContext:
    """Container for pipeline modules."""

    def __init__(self, settings: Settings) -> None:
        cam_cfg = settings.camera
        frame_shape = (cam_cfg.height, cam_cfg.width, 3)
        self.roi_gate = ROIGate((cam_cfg.height, cam_cfg.width), mode=settings.processing.roi_mode, expand_ratio=settings.processing.roi_expand)
        self.stabilizer = FrameStabilizer(frame_shape)
        self.background = BackgroundModel(frame_shape)
        self.detector = FmoLightDetector((cam_cfg.height, cam_cfg.width), threshold_factor=settings.processing.dfactor)
        self.color_prior = ColorPrior(ColorPriorConfig())
        self.tbd_refiner = TbDRefiner(TbDConfig())
        self.strike = StrikeZoneEvaluator(zone_min=(-0.2, -0.5, 0.5), zone_max=(0.2, 0.5, 1.2))
        self.camera_params = CameraParameters(fx=900.0, fy=900.0, cx=cam_cfg.width / 2, cy=cam_cfg.height / 2)
        self.settings = settings
        self.prev_center: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Strikezone FMO light pipeline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source", default=0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--exposure-sec", type=float, default=None)
    parser.add_argument("--roi-mode", choices=["expanded", "tight", "manual"], default="expanded")
    parser.add_argument("--dfactor", type=float, default=0.45)
    parser.add_argument("--enable-tbd", type=int, default=0)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    """Load settings from YAML and merge CLI overrides."""
    settings = load_settings(args.config)
    cam = settings.camera.copy()
    cam.source = args.source
    cam.width = args.width
    cam.height = args.height
    cam.fps = args.fps
    if args.exposure_sec is not None:
        cam.exposure_sec = args.exposure_sec
    proc = settings.processing.copy()
    proc.roi_mode = args.roi_mode
    proc.dfactor = args.dfactor
    proc.enable_tbd = bool(args.enable_tbd)
    out = settings.output.copy()
    out.save_path = args.save
    out.enable_debug = settings.output.enable_debug or False
    return Settings(camera=cam, processing=proc, output=out)


def configure_camera(capture: cv2.VideoCapture, config: CameraConfig) -> None:
    """Configure capture object with requested parameters."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)
    if config.exposure_sec is not None:
        capture.set(cv2.CAP_PROP_EXPOSURE, float(config.exposure_sec))


def run_loop(context: PipelineContext, capture: cv2.VideoCapture, writer: Optional[cv2.VideoWriter]) -> None:
    """Run the main acquisition and processing loop."""
    toggle = ToggleState()
    width = context.settings.camera.width
    height = context.settings.camera.height
    base_w = int(width * 0.45)
    base_h = int(height * 0.45)
    base_x = (width - base_w) // 2
    base_y = (height - base_h) // 2
    base_roi = ROIState(base_x, base_y, base_w, base_h)
    frame_time = 1.0 / max(context.settings.camera.fps, 1)

    while True:
        if toggle.paused:
            key = cv2.waitKey(5)
            if key == ord("q"):
                break
            if key == ord(" "):
                toggle.paused = False
            continue

        start = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            LOGGER.warning("Frame grab failed; exiting loop")
            break

        roi = context.roi_gate.compute_roi(base_roi)
        stab = context.stabilizer.stabilize(frame, roi.as_tuple())
        stable_frame = stab.frame
        gray = cv2.cvtColor(stable_frame, cv2.COLOR_BGR2GRAY)
        diff = context.background.apply(gray, roi.as_tuple())
        color_mask = context.color_prior.compute(stable_frame, roi.as_tuple())
        candidate = context.detector.detect(diff, roi.as_tuple(), color_mask)

        overlay = stable_frame.copy()
        vis.draw_roi(overlay, roi.as_tuple())
        decision_text = "No detection"
        if candidate:
            mask = candidate.mask
            if context.settings.processing.enable_tbd:
                mask = context.tbd_refiner.refine(mask)
            center = np.array(candidate.center, dtype=np.float32)
            displacement = (0.0, 0.0)
            if context.prev_center is not None:
                displacement = tuple((center - context.prev_center).tolist())
            trajectory = estimate_trajectory(
                center_px=tuple(center),
                displacement_px=displacement,
                params=context.camera_params,
                real_diameter_m=0.074,
                diameter_px=candidate.diameter_px,
                frame_time=frame_time,
            )
            decision = context.strike.evaluate(trajectory.position_m)
            decision_text = f"Strike" if decision.is_strike else "Ball"
            vis.draw_label(overlay, decision_text, (10, 30))
            vis.draw_velocity_vector(overlay, (int(center[0]), int(center[1])), displacement)
            context.prev_center = center
        else:
            context.prev_center = None

        if toggle.debug:
            vis.draw_label(overlay, decision_text, (10, 50), (0, 255, 255))

        if toggle.color_preview:
            roi_mask = np.zeros((overlay.shape[0], overlay.shape[1]), dtype=np.uint8)
            x, y, w, h = roi.as_tuple()
            roi_mask[y : y + h, x : x + w] = color_mask
            vis.overlay_mask(overlay, roi_mask)

        cv2.imshow("Strikezone", overlay if not toggle.background_preview else diff)
        if writer is not None:
            writer.write(overlay)

        elapsed = time.perf_counter() - start
        fps = 1.0 / max(elapsed, 1e-6)
        LOGGER.debug("Frame processed in %.2f ms (%.1f FPS)", elapsed * 1000, fps)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            toggle.paused = True
        if key == ord("d"):
            toggle.debug = not toggle.debug
        if key == ord("c"):
            toggle.color_preview = not toggle.color_preview
        if key == ord("b"):
            toggle.background_preview = not toggle.background_preview
        if key == ord("r"):
            context.roi_gate.cycle_mode()

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    """Entry point executed via ``python -m src.main``."""
    args = parse_args()
    configure_logging(getattr(__import__("logging"), args.log_level))
    settings = build_settings(args)

    source = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        LOGGER.warning("Failed to open video source %s", source)
    configure_camera(capture, settings.camera)

    writer = None
    if settings.output.save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(settings.output.save_path), fourcc, settings.camera.fps, (settings.camera.width, settings.camera.height))

    context = PipelineContext(settings)
    try:
        run_loop(context, capture, writer)
    finally:
        if capture.isOpened():
            capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

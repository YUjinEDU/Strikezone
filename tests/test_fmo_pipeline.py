import numpy as np
from scipy import ndimage as ndi

from pathlib import Path

from strikezone.config import CameraIntrinsics, load_config
from strikezone.pipeline import StrikeZonePipeline


def _render_motion_blur(height, width, start, end, radius, steps=7):
    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for t in np.linspace(0.0, 1.0, steps):
        center = start + (end - start) * t
        dist = np.sqrt((yy - center[1]) ** 2 + (xx - center[0]) ** 2)
        frame = np.ones((height, width), dtype=np.float32)
        frame[dist <= radius] = 0.1
        frames.append(frame)
    blurred = np.mean(frames, axis=0)
    blurred = ndi.gaussian_filter(blurred, sigma=0.5)
    return blurred


def _generate_sequence(num_frames=20, height=64, width=64):
    centers = []
    frames = []
    radius = 5.0
    for i in range(num_frames):
        start = np.array([10.0 + i * 1.5, height / 2])
        end = start + np.array([2.0, 0.0])
        centers.append(start + (end - start) * 0.5)
        frame = _render_motion_blur(height, width, start, end, radius)
        frames.append((frame * 255).astype(np.uint8))
    return np.stack(frames), np.stack(centers)


def test_pipeline_recall_exceeds_threshold():
    frames, centers = _generate_sequence()

    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(config_path)

    intrinsics = CameraIntrinsics(fx=850.0, fy=850.0, cx=32.0, cy=32.0)
    pipeline = StrikeZonePipeline(intrinsics, config)

    candidates = pipeline.detect(frames)
    detections_by_frame = {}
    for cand in candidates:
        detections_by_frame.setdefault(cand.frame_index, []).append(cand)

    hits = 0
    total = len(centers)
    for idx, center in enumerate(centers):
        detections = detections_by_frame.get(idx, [])
        if not detections:
            continue
        best = min(detections, key=lambda c: np.linalg.norm(c.center_uv - center))
        if np.linalg.norm(best.center_uv - center) <= 6.0:
            hits += 1
    recall = hits / total

    assert recall >= 0.9

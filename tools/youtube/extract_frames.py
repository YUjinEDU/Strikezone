import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class FrameIndexEntry:
    frame_idx: int
    t_sec: float
    filename: str


@dataclass(frozen=True)
class ClipIndex:
    clip_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    extracted_every_n: int
    frames: List[FrameIndexEntry]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_video_file(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _safe_stem(stem: str) -> str:
    # keep it stable for folder names
    return "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in stem)


def extract_frames_from_clip(
    clip_path: Path,
    out_dir: Path,
    *,
    every_n: int = 1,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 95,
) -> ClipIndex:
    if every_n < 1:
        raise ValueError("every_n must be >= 1")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames_out: List[FrameIndexEntry] = []
    saved = 0
    idx = 0

    # Ensure output directory exists
    _ensure_dir(out_dir)

    # OpenCV JPEG quality param
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, jpeg_quality)))]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % every_n == 0:
            # timestamp: prefer POS_MSEC, fall back to idx/fps
            t_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if fps else (idx * 0.0)
            if fps and (t_sec == 0.0 or math.isnan(t_sec)):
                t_sec = idx / fps

            filename = f"frame_{idx:06d}.jpg"
            out_path = out_dir / filename
            ok = cv2.imwrite(str(out_path), frame, jpeg_params)
            if not ok:
                raise RuntimeError(f"Failed to write frame: {out_path}")

            frames_out.append(FrameIndexEntry(frame_idx=idx, t_sec=float(t_sec), filename=filename))
            saved += 1

            if max_frames is not None and saved >= max_frames:
                break

        idx += 1

    cap.release()

    index = ClipIndex(
        clip_path=str(clip_path.as_posix()),
        width=width,
        height=height,
        fps=float(fps),
        frame_count=int(frame_count),
        extracted_every_n=int(every_n),
        frames=frames_out,
    )
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract frames (jpg) from downloaded YouTube clip files.")
    ap.add_argument("--clips_dir", default="data/youtube/clips", help="Directory containing clip mp4 files")
    ap.add_argument("--out_dir", default="data/youtube/frames", help="Output directory for extracted frames")
    ap.add_argument("--every_n", type=int, default=1, help="Extract every Nth frame (1 = all frames)")
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames per clip (0 = no limit)")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100)")
    ap.add_argument("--index_name", default="index.json", help="Index file name per clip folder")
    ap.add_argument("--skip_existing", action="store_true", help="Skip clip if index.json already exists")
    ap.add_argument(
        "--delete_clips",
        action="store_true",
        help="Delete source clip file after successful extraction (saves disk space).",
    )
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir)
    out_root = Path(args.out_dir)
    _ensure_dir(out_root)

    if not clips_dir.exists():
        print(f"clips_dir not found: {clips_dir}")
        return 2

    max_frames = None if args.max_frames <= 0 else int(args.max_frames)

    clip_files = [p for p in clips_dir.iterdir() if p.is_file() and _is_video_file(p)]
    clip_files.sort(key=lambda p: p.name)

    if not clip_files:
        print(f"No clip files found in: {clips_dir}")
        return 1

    ok = 0
    fail = 0

    for clip_path in clip_files:
        clip_folder = out_root / _safe_stem(clip_path.stem)
        _ensure_dir(clip_folder)
        index_path = clip_folder / args.index_name

        if args.skip_existing and index_path.exists():
            print(f"SKIP: {clip_path.name} (index exists)")
            continue

        try:
            index = extract_frames_from_clip(
                clip_path,
                clip_folder,
                every_n=int(args.every_n),
                max_frames=max_frames,
                jpeg_quality=int(args.jpeg_quality),
            )
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        **{k: v for k, v in asdict(index).items() if k != "frames"},
                        "frames": [asdict(x) for x in index.frames],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"OK: {clip_path.name} -> {clip_folder}")
            ok += 1

            if args.delete_clips:
                try:
                    clip_path.unlink()
                    print(f"DELETED: {clip_path.name}")
                except Exception as e:
                    print(f"WARN: failed to delete clip {clip_path.name}: {e}")
        except Exception as e:
            print(f"FAIL: {clip_path.name}: {e}")
            fail += 1

    print(f"Done. ok={ok}, fail={fail}")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())



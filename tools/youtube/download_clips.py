import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def _extract_video_id(url: str) -> Optional[str]:
    m = YOUTUBE_ID_RE.search(url)
    if not m:
        return None
    return m.group(1)


def _sec_to_ms(sec: float) -> int:
    return int(round(float(sec) * 1000))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_section(start_sec: float, end_sec: float) -> str:
    # yt-dlp --download-sections accepts: "*start-end"
    # we keep seconds with millisecond precision
    return f"*{start_sec:.3f}-{end_sec:.3f}"


def _safe_slug(s: str) -> str:
    # conservative: only keep alnum, underscore, dash
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def _build_format_selector(base_format: str, min_fps: int, min_height: int) -> str:
    """
    Build a robust yt-dlp format selector that prefers the highest FPS and quality.

    If base_format != 'auto', return it as-is.
    If base_format == 'auto', try strict constraints first, then fall back progressively:
      1) fps>=min_fps AND height>=min_height
      2) fps>=min_fps
      3) height>=min_height
      4) anything bestvideo+bestaudio
    """
    if base_format.strip().lower() != "auto":
        return base_format

    fps_part = f"[fps>={int(min_fps)}]" if int(min_fps) > 0 else ""
    h_part = f"[height>={int(min_height)}]" if int(min_height) > 0 else ""

    candidates: List[str] = []
    if fps_part and h_part:
        candidates.append(f"bestvideo*{fps_part}{h_part}+bestaudio/best")
    if fps_part:
        candidates.append(f"bestvideo*{fps_part}+bestaudio/best")
    if h_part:
        candidates.append(f"bestvideo*{h_part}+bestaudio/best")
    candidates.append("bestvideo*+bestaudio/best")
    return "/".join(candidates)


@dataclass(frozen=True)
class ClipJob:
    dataset_id: str
    view_type: str
    video_id: str
    url: str
    clip_id: str
    start_sec: float
    end_sec: float

    @property
    def start_ms(self) -> int:
        return _sec_to_ms(self.start_sec)

    @property
    def end_ms(self) -> int:
        return _sec_to_ms(self.end_sec)

    @property
    def filename(self) -> str:
        # {dataset_id}__{view_type}__{videoId}__{clip_id}__s{start_ms}__e{end_ms}.mp4
        return (
            f"{_safe_slug(self.dataset_id)}__{_safe_slug(self.view_type)}__{self.video_id}"
            f"__{_safe_slug(self.clip_id)}__s{self.start_ms}__e{self.end_ms}.mp4"
        )


def _collect_jobs(manifest: Dict[str, Any]) -> List[ClipJob]:
    dataset_id = manifest.get("dataset_id") or "dataset"
    default_view_type = None
    default_obj = manifest.get("default")
    if isinstance(default_obj, dict) and isinstance(default_obj.get("view_type"), str):
        default_view_type = default_obj.get("view_type")

    jobs: List[ClipJob] = []
    videos = manifest.get("videos") or []
    for video in videos:
        url = video.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()

        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            extracted = _extract_video_id(url)
            if not extracted:
                raise ValueError(f"Could not extract video_id from url: {url}")
            video_id = extracted
        else:
            video_id = video_id.strip()

        view_type = video.get("view_type") or default_view_type or "unknown"
        clips = video.get("clips") or []
        for clip in clips:
            clip_id = clip.get("clip_id")
            start_sec = clip.get("start_sec")
            end_sec = clip.get("end_sec")
            if not isinstance(clip_id, str) or not clip_id.strip():
                continue
            if start_sec is None or end_sec is None:
                continue
            jobs.append(
                ClipJob(
                    dataset_id=str(dataset_id),
                    view_type=str(view_type),
                    video_id=str(video_id),
                    url=url,
                    clip_id=str(clip_id),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                )
            )
    return jobs


def _run(cmd: List[str], log_path: Path) -> int:
    _ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n")
        log.write(f"=== {dt.datetime.now().isoformat()} ===\n")
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=log, text=True)
        return p.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download YouTube pitching segments defined in a manifest JSON.")
    ap.add_argument("--manifest", required=True, help="Path to manifest JSON")
    ap.add_argument("--out_dir", default="data/youtube/clips", help="Output directory for clips")
    ap.add_argument("--log_dir", default="data/youtube/logs", help="Directory for logs")
    ap.add_argument(
        "--format",
        default="auto",
        help="yt-dlp format selector. Use 'auto' to prefer highest FPS/quality with fallbacks.",
    )
    ap.add_argument(
        "--min_fps",
        type=int,
        default=60,
        help="Minimum FPS to prefer when --format=auto (0 disables FPS constraint).",
    )
    ap.add_argument(
        "--min_height",
        type=int,
        default=1080,
        help="Minimum height to prefer when --format=auto (0 disables height constraint).",
    )
    ap.add_argument(
        "--format_sort",
        default="fps,res,br",
        help="yt-dlp format sorting keys (-S). Default prefers FPS then resolution then bitrate.",
    )
    ap.add_argument("--retries", type=int, default=2, help="Retry count per clip on failure")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output file already exists")
    ap.add_argument("--yt_dlp", default="yt-dlp", help="yt-dlp executable (in PATH)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (in PATH)")
    ap.add_argument("--force_mp4", action="store_true", help="Force mp4 container for output")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    _ensure_dir(out_dir)
    _ensure_dir(log_dir)

    manifest = _read_json(manifest_path)
    jobs = _collect_jobs(manifest)
    if not jobs:
        print("No clip jobs found in manifest.")
        return 1

    dataset_id = str(manifest.get("dataset_id") or "dataset")
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log = log_dir / f"download_{_safe_slug(dataset_id)}_{run_id}.log"

    format_selector = _build_format_selector(str(args.format), int(args.min_fps), int(args.min_height))

    ok = 0
    fail = 0

    for job in jobs:
        out_path = out_dir / job.filename
        if args.skip_existing and out_path.exists():
            print(f"SKIP: {out_path.name} (exists)")
            continue

        section = _format_section(job.start_sec, job.end_sec)

        # yt-dlp notes:
        # - --download-sections requires ffmpeg for accurate cutting/merging.
        # - -o should include extension template; we force exact filename with -o.
        # - to enforce mp4, we set --merge-output-format mp4 when requested.
        cmd = [
            args.yt_dlp,
            "--no-playlist",
            "--download-sections",
            section,
            "-f",
            format_selector,
            "-S",
            str(args.format_sort),
            "--ffmpeg-location",
            args.ffmpeg,
            "--newline",
            "--no-progress",
            "-o",
            str(out_path),
        ]
        if args.force_mp4:
            cmd += ["--merge-output-format", "mp4"]

        attempt = 0
        last_code: Optional[int] = None
        while attempt <= args.retries:
            attempt += 1
            print(f"DOWNLOADING: {job.video_id} {job.clip_id} {job.start_sec:.3f}-{job.end_sec:.3f}s -> {out_path.name}")
            code = _run(cmd, run_log)
            last_code = code
            if code == 0 and out_path.exists() and out_path.stat().st_size > 0:
                ok += 1
                break
            else:
                if attempt <= args.retries:
                    print(f"RETRY({attempt}/{args.retries}) for {out_path.name} (exit={code})")
                else:
                    print(f"FAILED: {out_path.name} (exit={code})")
                    fail += 1

        # safety: if failed, keep going
        _ = last_code

    print(f"Done. ok={ok}, fail={fail}, log={run_log}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())



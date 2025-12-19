import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _extract_video_id(url: str) -> Optional[str]:
    m = YOUTUBE_ID_RE.search(url)
    if not m:
        return None
    return m.group(1)


def _sec_to_ms(sec: float) -> int:
    return int(round(sec * 1000))


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "ERROR" | "WARN"
    where: str
    message: str


def validate_manifest(manifest: Dict[str, Any]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        issues.append(ValidationIssue("ERROR", "root.dataset_id", "dataset_id (string) is required"))

    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) == 0:
        issues.append(ValidationIssue("ERROR", "root.videos", "videos must be a non-empty array"))
        return issues

    default_view_type = None
    default_obj = manifest.get("default")
    if isinstance(default_obj, dict):
        dv = default_obj.get("view_type")
        if isinstance(dv, str) and dv.strip():
            default_view_type = dv.strip()

    for vi, video in enumerate(videos):
        where_video = f"videos[{vi}]"
        if not isinstance(video, dict):
            issues.append(ValidationIssue("ERROR", where_video, "video entry must be an object"))
            continue

        url = video.get("url")
        if not isinstance(url, str) or not url.strip():
            issues.append(ValidationIssue("ERROR", f"{where_video}.url", "url (string) is required"))
            continue

        video_id = video.get("video_id")
        extracted_id = _extract_video_id(url)
        if not isinstance(video_id, str) or not video_id.strip():
            if extracted_id is None:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        f"{where_video}.video_id",
                        "video_id missing and could not extract from url; add video_id explicitly",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        "WARN",
                        f"{where_video}.video_id",
                        f"video_id missing; can be auto-extracted as {extracted_id}",
                    )
                )
        else:
            if extracted_id and video_id.strip() != extracted_id:
                issues.append(
                    ValidationIssue(
                        "WARN",
                        f"{where_video}.video_id",
                        f"video_id ({video_id}) differs from url-extracted id ({extracted_id})",
                    )
                )

        view_type = video.get("view_type") or default_view_type
        if not isinstance(view_type, str) or not view_type.strip():
            issues.append(
                ValidationIssue(
                    "WARN",
                    f"{where_video}.view_type",
                    "view_type missing (will impact naming); set video.view_type or default.view_type",
                )
            )

        clips = video.get("clips")
        if not isinstance(clips, list) or len(clips) == 0:
            issues.append(ValidationIssue("ERROR", f"{where_video}.clips", "clips must be a non-empty array"))
            continue

        seen_clip_ids: set[str] = set()
        intervals: List[Tuple[float, float, str]] = []

        for ci, clip in enumerate(clips):
            where_clip = f"{where_video}.clips[{ci}]"
            if not isinstance(clip, dict):
                issues.append(ValidationIssue("ERROR", where_clip, "clip entry must be an object"))
                continue

            clip_id = clip.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id.strip():
                issues.append(ValidationIssue("ERROR", f"{where_clip}.clip_id", "clip_id (string) is required"))
                continue

            if clip_id in seen_clip_ids:
                issues.append(
                    ValidationIssue("ERROR", f"{where_clip}.clip_id", f"duplicate clip_id within video: {clip_id}")
                )
            seen_clip_ids.add(clip_id)

            start_sec = clip.get("start_sec")
            end_sec = clip.get("end_sec")
            if not _is_number(start_sec):
                issues.append(ValidationIssue("ERROR", f"{where_clip}.start_sec", "start_sec must be a number"))
                continue
            if not _is_number(end_sec):
                issues.append(ValidationIssue("ERROR", f"{where_clip}.end_sec", "end_sec must be a number"))
                continue

            start_sec = float(start_sec)
            end_sec = float(end_sec)

            if start_sec < 0 or end_sec < 0:
                issues.append(
                    ValidationIssue("ERROR", where_clip, f"start/end must be >= 0 (got {start_sec}, {end_sec})")
                )
                continue

            if not (start_sec < end_sec):
                issues.append(
                    ValidationIssue("ERROR", where_clip, f"require start_sec < end_sec (got {start_sec}, {end_sec})")
                )
                continue

            dur = end_sec - start_sec
            if dur < 0.15:
                issues.append(
                    ValidationIssue(
                        "WARN",
                        where_clip,
                        f"clip duration is very short ({dur:.3f}s). Is end_sec correct?",
                    )
                )
            if dur > 5.0:
                issues.append(
                    ValidationIssue(
                        "WARN",
                        where_clip,
                        f"clip duration is long ({dur:.2f}s). For pitching segments, verify start/end.",
                    )
                )

            start_ms = _sec_to_ms(start_sec)
            end_ms = _sec_to_ms(end_sec)
            if end_ms <= start_ms:
                issues.append(
                    ValidationIssue(
                        "ERROR",
                        where_clip,
                        f"ms rounding made invalid interval: start_ms={start_ms}, end_ms={end_ms}",
                    )
                )
                continue

            intervals.append((start_sec, end_sec, clip_id))

        # overlap detection (WARN)
        intervals.sort(key=lambda x: (x[0], x[1]))
        for i in range(1, len(intervals)):
            prev_s, prev_e, prev_id = intervals[i - 1]
            cur_s, cur_e, cur_id = intervals[i]
            if cur_s < prev_e:
                issues.append(
                    ValidationIssue(
                        "WARN",
                        where_video,
                        f"overlapping clips: {prev_id} ({prev_s:.3f}-{prev_e:.3f}) overlaps {cur_id} ({cur_s:.3f}-{cur_e:.3f})",
                    )
                )

    return issues


def _print_issues(issues: List[ValidationIssue]) -> None:
    if not issues:
        print("OK: manifest is valid")
        return

    width_level = max(len(i.level) for i in issues)
    width_where = min(80, max(len(i.where) for i in issues))

    for i in issues:
        where = i.where if len(i.where) <= width_where else (i.where[: width_where - 3] + "...")
        print(f"{i.level:<{width_level}}  {where:<{width_where}}  {i.message}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate YouTube dataset manifest JSON.")
    ap.add_argument("--manifest", required=True, help="Path to manifest JSON")
    ap.add_argument("--strict", action="store_true", help="Treat WARN as ERROR (non-zero exit)")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"ERROR: file not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = _read_json(args.manifest)
    issues = validate_manifest(manifest)
    _print_issues(issues)

    has_error = any(i.level == "ERROR" for i in issues)
    has_warn = any(i.level == "WARN" for i in issues)

    if has_error:
        return 1
    if args.strict and has_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



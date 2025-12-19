import argparse
import json
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Pair:
    image_path: str
    label_path: str
    clip_group: str  # parent folder name (group id)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _find_pairs(frames_root: Path, labels_ext: str = ".txt") -> List[Pair]:
    """
    Recursively find image/label pairs under frames_root.

    Expected:
      - image: *.jpg|*.png
      - label: same stem with .txt in same folder
    """
    pairs: List[Pair] = []
    for img in frames_root.rglob("*"):
        if not _is_image(img):
            continue
        lbl = img.with_suffix(labels_ext)
        if not lbl.exists():
            continue
        group = img.parent.name
        pairs.append(Pair(image_path=str(img), label_path=str(lbl), clip_group=group))
    return pairs


def _split_groups(groups: List[str], val_ratio: float, test_ratio: float, rng: random.Random) -> Tuple[set, set, set]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio+test_ratio < 1")

    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test = set(groups[:n_test])
    val = set(groups[n_test : n_test + n_val])
    train = set(groups[n_test + n_val :])
    return train, val, test


def _copy_or_move(src: Path, dst: Path, move: bool) -> None:
    _ensure_dir(dst.parent)
    if move:
        # ensure overwrite semantics (Windows move fails if exists)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split labeled frames into YOLO dataset structure (train/val/test)."
    )
    ap.add_argument(
        "--frames_root",
        default="data/youtube/frames",
        help="Root directory containing extracted frames (and YOLO label .txt next to images)",
    )
    ap.add_argument("--dataset_dir", default="dataset/frames", help="Target dataset/frames directory")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (by clip group)")
    ap.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (by clip group)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--by",
        choices=["clip", "frame"],
        default="clip",
        help="Split granularity: clip (recommended) or frame (not recommended)",
    )
    ap.add_argument("--move", action="store_true", help="Move files instead of copying (destructive)")
    ap.add_argument("--dry_run", action="store_true", help="Do not copy/move, only print plan")
    ap.add_argument("--report", default="split_report.json", help="Report filename written under dataset_dir")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    dataset_dir = Path(args.dataset_dir)

    if not frames_root.exists():
        print(f"frames_root not found: {frames_root}")
        return 2

    pairs = _find_pairs(frames_root)
    if not pairs:
        print(f"No image+label pairs found under: {frames_root}")
        print("Tip: label files must exist next to images with same stem and .txt extension.")
        return 1

    rng = random.Random(args.seed)

    if args.by == "clip":
        group_ids = sorted({p.clip_group for p in pairs})
        train_g, val_g, test_g = _split_groups(group_ids, args.val_ratio, args.test_ratio, rng)
        split_for_pair = {
            "train": train_g,
            "val": val_g,
            "test": test_g,
        }

        def which_split(p: Pair) -> str:
            for s, gs in split_for_pair.items():
                if p.clip_group in gs:
                    return s
            return "train"

    else:
        # frame-level split
        idxs = list(range(len(pairs)))
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * args.test_ratio))
        n_val = int(round(n * args.val_ratio))
        test_i = set(idxs[:n_test])
        val_i = set(idxs[n_test : n_test + n_val])

        def which_split(p: Pair) -> str:
            # stable index lookup based on path
            # build map once
            return "train"

        # build a map for frame-level split
        path_to_split: Dict[str, str] = {}
        for j, idx in enumerate(idxs):
            s = "train"
            if j < n_test:
                s = "test"
            elif j < n_test + n_val:
                s = "val"
            path_to_split[pairs[idx].image_path] = s

        def which_split(p: Pair) -> str:
            return path_to_split.get(p.image_path, "train")

    targets = {
        "train": {
            "images": dataset_dir / "images" / "train",
            "labels": dataset_dir / "labels" / "train",
        },
        "val": {
            "images": dataset_dir / "images" / "val",
            "labels": dataset_dir / "labels" / "val",
        },
        "test": {
            "images": dataset_dir / "images" / "test",
            "labels": dataset_dir / "labels" / "test",
        },
    }
    for s in targets:
        _ensure_dir(targets[s]["images"])
        _ensure_dir(targets[s]["labels"])

    counts = {"train": 0, "val": 0, "test": 0}
    moved_or_copied: List[Dict[str, str]] = []

    for p in pairs:
        split = which_split(p)
        img_src = Path(p.image_path)
        lbl_src = Path(p.label_path)

        # keep filenames unique by prefixing clip group when needed
        # (prevent collisions if different clips have same frame_000001.jpg)
        dst_base = f"{p.clip_group}__{img_src.name}"
        img_dst = targets[split]["images"] / dst_base
        lbl_dst = targets[split]["labels"] / (Path(dst_base).with_suffix(".txt").name)

        if args.dry_run:
            counts[split] += 1
            continue

        _copy_or_move(img_src, img_dst, args.move)
        _copy_or_move(lbl_src, lbl_dst, args.move)
        counts[split] += 1
        moved_or_copied.append(
            {
                "split": split,
                "image_dst": str(img_dst),
                "label_dst": str(lbl_dst),
                "clip_group": p.clip_group,
            }
        )

    report = {
        "frames_root": str(frames_root),
        "dataset_dir": str(dataset_dir),
        "mode": "move" if args.move else "copy",
        "by": args.by,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": counts,
        "pairs_found": len(pairs),
    }

    if args.dry_run:
        print("DRY RUN")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    report_path = dataset_dir / args.report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Done. train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



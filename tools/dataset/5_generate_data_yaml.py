import argparse
from pathlib import Path


DEFAULT_NAMES = ["baseball"]


def _detect_path_prefix(dataset_root: Path, use_absolute: bool) -> str:
    if use_absolute:
        return str(dataset_root.resolve())
    # keep yaml portable: relative path from repo root
    return str(dataset_root.as_posix())


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate YOLO data.yaml for Ultralytics training.")
    ap.add_argument(
        "--dataset_root",
        default="dataset",
        help="Dataset root directory that contains frames/images/* and frames/labels/*",
    )
    ap.add_argument(
        "--out",
        default="dataset/data.yaml",
        help="Output yaml path (default: dataset/data.yaml)",
    )
    ap.add_argument("--names", default="baseball", help="Comma-separated class names (default: baseball)")
    ap.add_argument("--use_absolute_path", action="store_true", help="Write absolute dataset path in yaml")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    frames_dir = dataset_root / "frames"

    images_train = frames_dir / "images" / "train"
    images_val = frames_dir / "images" / "val"
    images_test = frames_dir / "images" / "test"

    # We don't hard-require test
    if not images_train.exists() or not images_val.exists():
        print("ERROR: expected dataset structure not found. Run:")
        print("  python tools/dataset/3_init_yolo_dataset.py")
        return 2

    names = [x.strip() for x in str(args.names).split(",") if x.strip()]
    if not names:
        names = DEFAULT_NAMES

    nc = len(names)
    path_prefix = _detect_path_prefix(dataset_root, args.use_absolute_path)

    # Ultralytics accepts relative paths from 'path'
    yaml = (
        f"path: {path_prefix}\n"
        f"train: frames/images/train\n"
        f"val: frames/images/val\n"
        f"test: frames/images/test\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )

    out_path = Path(args.out)
    _write_yaml(out_path, yaml)
    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



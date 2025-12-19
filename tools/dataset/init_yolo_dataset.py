import argparse
from pathlib import Path


DEFAULT_STRUCTURE = [
    "dataset/frames/images/train",
    "dataset/frames/images/val",
    "dataset/frames/images/test",
    "dataset/frames/labels/train",
    "dataset/frames/labels/val",
    "dataset/frames/labels/test",
    "dataset/metadata",
    "dataset/benchmarks",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize YOLO-style dataset folder skeleton.")
    ap.add_argument("--root", default=".", help="Project root (default: current dir)")
    ap.add_argument("--dataset_dir", default="dataset", help="Dataset directory name (default: dataset)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = args.dataset_dir.strip("/").strip("\\")

    for rel in DEFAULT_STRUCTURE:
        rel2 = rel.replace("dataset", dataset_dir, 1)
        ensure_dir(root / rel2)

    # Keep empty dirs in git if user wants; but dataset can be large so we avoid adding gitkeep here.
    print(f"Initialized dataset skeleton at: {root / dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



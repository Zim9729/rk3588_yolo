"""Scan an image directory and write a calibration dataset.txt for RKNN quantization."""
import argparse
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate calibration dataset.txt from an image directory.")
    parser.add_argument("--input", required=True, help="Directory containing calibration images.")
    parser.add_argument("--output", default="data/calibration/dataset.txt", help="Output dataset.txt path.")
    parser.add_argument("--count", type=int, default=300, help="Max number of images to include (0 = all).")
    parser.add_argument("--wsl", action="store_true", help="Convert Windows paths to WSL /mnt/ format.")
    parser.add_argument("--shuffle", action="store_true", help="Randomly sample instead of taking the first N.")
    return parser.parse_args()


def to_wsl_path(p: Path) -> str:
    """E:\\foo\\bar -> /mnt/e/foo/bar"""
    drive = p.drive[0].lower()  # 'E' -> 'e'
    return f"/mnt/{drive}" + str(p).replace(p.drive, "").replace("\\", "/")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise ValueError(f"No images found in: {input_dir}")

    if args.count > 0 and len(images) > args.count:
        if args.shuffle:
            images = random.sample(images, args.count)
        else:
            images = images[: args.count]

    if args.wsl:
        paths = [to_wsl_path(p) for p in images]
    else:
        paths = [str(p) for p in images]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")

    print(f"Images found: {len(images)} (from {input_dir})")
    print(f"Written: {output} ({len(paths)} paths)")
    if paths:
        print(f"  First: {paths[0]}")
        print(f"  Last:  {paths[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

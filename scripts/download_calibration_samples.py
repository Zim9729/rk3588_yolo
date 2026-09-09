"""Download sample images for quick quantization calibration validation."""
import os
import urllib.request
from pathlib import Path


def download_samples(output_dir: Path, count: int = 15) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        filename = output_dir / f"sample_{i:03d}.jpg"
        if filename.exists():
            print(f"  Skip existing: {filename}")
            continue
        url = f"https://picsum.photos/640/640.jpg?random={i}"
        print(f"  Downloading: {filename.name} ...")
        urllib.request.urlretrieve(url, str(filename))

    dataset_txt = output_dir / "dataset.txt"
    dataset_txt.write_text(
        "\n".join(p.name for p in sorted(output_dir.glob("*.jpg"))) + "\n",
        encoding="utf-8",
    )
    print(f"Done. {count} images saved to {output_dir}")
    print(f"Dataset list: {dataset_txt}")


if __name__ == "__main__":
    download_samples(Path("data/calibration"), count=15)

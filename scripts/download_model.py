"""
Download pretrained nnUNet Dataset137 (BraTS-2021) checkpoint files from Google Drive.

Files downloaded:
  - checkpoint_final.pth (fold_5 weights)
  - plans.json
  - dataset.json

Output structure:
  <output_dir>/
    nnUNetTrainer__nnUNetPlans__3d_fullres/
      fold_5/
        checkpoint_final.pth
        plans.json
        dataset.json
    dataset.json   <- copy of the above
"""

import argparse
import os
import shutil
from pathlib import Path

import gdown

GDRIVE_IDS = {
    "checkpoint_final.pth": "1n9dqT114udr9Qq8iYEKsJK347iHg9N88",
    "dataset.json": "1A_suxQwElucF3w1HEYg3wMo6dG9OxBHo",
    "plans.json": "1U2b0BTNi8zrJACReoi_W08Fe-wM394wI",
}


def download_file(file_id: str, dest_path: Path) -> None:
    """Download a single file from Google Drive if not already present."""
    if dest_path.exists():
        print(f"  [skip] {dest_path.name} already exists ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {dest_path.name} ...")
    gdown.download(url, str(dest_path), quiet=False)
    print(f"  Done: {dest_path.stat().st_size / 1e6:.1f} MB")


def build_directory_structure(output_dir: Path) -> tuple[Path, Path]:
    """Create the nnUNet directory layout and return (fold_dir, top_dir)."""
    fold_dir = output_dir / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "fold_5"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dataset137 (BraTS-2021 pretrained) nnUNet checkpoint files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./models/Dataset137_BraTS2021"),
        help="Root directory to save model files (default: ./models/Dataset137_BraTS2021).",
    )
    args = parser.parse_args()

    fold_dir, top_dir = build_directory_structure(args.output_dir)
    print(f"Output directory: {args.output_dir.resolve()}")

    # Download checkpoint into fold_5/
    download_file(GDRIVE_IDS["checkpoint_final.pth"], fold_dir / "checkpoint_final.pth")

    # Download plans.json and dataset.json into fold_5/
    for name in ("plans.json", "dataset.json"):
        download_file(GDRIVE_IDS[name], fold_dir / name)

    # Copy dataset.json to top-level directory as nnUNet expects it there too
    top_level_dataset = top_dir / "dataset.json"
    if not top_level_dataset.exists():
        shutil.copy2(fold_dir / "dataset.json", top_level_dataset)
        print(f"  Copied dataset.json → {top_level_dataset}")
    else:
        print(f"  [skip] top-level dataset.json already exists")

    print("\nDownload complete. File summary:")
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(args.output_dir)
            print(f"  {rel}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

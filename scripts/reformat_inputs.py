"""
Reformat BraTS 2023 raw data into nnUNet imagesTs/ format for inference.

Works with any split (train/val/test) and uses symlinks to avoid duplicating data.

Input layout:
  <data_dir>/<split>/{case}/{case}-t1n.nii.gz
                            {case}-t1c.nii.gz
                            {case}-t2w.nii.gz
                            {case}-t2f.nii.gz

Output layout:
  <output_dir>/
    {case}_0000.nii.gz  <- t1n
    {case}_0001.nii.gz  <- t1c
    {case}_0002.nii.gz  <- t2w
    {case}_0003.nii.gz  <- t2f
"""

import argparse
from pathlib import Path

MODALITY_MAP = {
    "_0000": "-t1n",
    "_0001": "-t1c",
    "_0002": "-t2w",
    "_0003": "-t2f",
}


def reformat_case(case_dir: Path, output_dir: Path) -> bool:
    """
    Create nnUNet-style symlinks for one subject's four modality files.

    Returns True on success, False if any modality file is missing.
    """
    case = case_dir.name
    missing = []

    for suffix, mod in MODALITY_MAP.items():
        src = case_dir / f"{case}{mod}.nii.gz"
        if not src.exists():
            missing.append(src.name)

    if missing:
        print(f"  [warn] Skipping {case}: missing {', '.join(missing)}")
        return False

    for suffix, mod in MODALITY_MAP.items():
        src = (case_dir / f"{case}{mod}.nii.gz").resolve()
        dst = output_dir / f"{case}{suffix}.nii.gz"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat BraTS 2023 raw data to nnUNet imagesTs/ layout using symlinks."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root of the BraTS 2023 raw dataset (contains train/, val/, test/ subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/pvc/nnunet/imagesTs"),
        help="Destination directory for nnUNet-formatted inputs (default: /pvc/nnunet/imagesTs).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which BraTS 2023 split to reformat (default: test).",
    )
    args = parser.parse_args()

    split_dir = args.data_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in split_dir.iterdir() if p.is_dir())
    print(f"Found {len(cases)} subjects in {split_dir}")

    processed, skipped = 0, 0
    for case_dir in cases:
        if reformat_case(case_dir, args.output_dir):
            processed += 1
        else:
            skipped += 1

    print(f"\nDone: {processed} subjects reformatted, {skipped} skipped.")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

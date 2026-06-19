"""
Validate BraTS 2023 folder structure and NIfTI file integrity.

For each subject directory, checks:
  1. All 5 expected files are present (t1n, t1c, t2w, t2f, seg).
  2. Each .nii.gz can be opened and has a non-empty volume with nibabel.

Prints a pass/fail summary with counts.
"""

import argparse
from pathlib import Path

import nibabel as nib

EXPECTED_SUFFIXES = ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]


def check_subject(case_dir: Path) -> tuple[list[str], list[str]]:
    """
    Check one subject directory for missing files and load errors.

    Returns (missing_files, corrupt_files).
    """
    case = case_dir.name
    missing = []
    corrupt = []

    for suffix in EXPECTED_SUFFIXES:
        fpath = case_dir / f"{case}{suffix}"
        if not fpath.exists():
            missing.append(fpath.name)
            continue
        try:
            img = nib.load(str(fpath))
            shape = img.shape
            if any(s == 0 for s in shape):
                corrupt.append(f"{fpath.name} (empty shape {shape})")
        except Exception as exc:
            corrupt.append(f"{fpath.name} ({exc})")

    return missing, corrupt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate BraTS 2023 dataset folder structure and NIfTI file integrity."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root of the BraTS 2023 raw dataset (contains train/, val/, test/ subdirs).",
    )
    args = parser.parse_args()

    splits = [d for d in ("train", "val", "test") if (args.data_dir / d).exists()]
    if not splits:
        raise FileNotFoundError(
            f"No split directories (train/val/test) found under {args.data_dir}"
        )

    total_subjects = 0
    total_pass = 0
    total_missing_files = 0
    total_corrupt_files = 0

    for split in splits:
        split_dir = args.data_dir / split
        cases = sorted(p for p in split_dir.iterdir() if p.is_dir())
        print(f"\n[{split}] {len(cases)} subjects")

        split_pass = 0
        for case_dir in cases:
            missing, corrupt = check_subject(case_dir)
            if missing or corrupt:
                print(f"  FAIL  {case_dir.name}")
                for f in missing:
                    print(f"        missing:  {f}")
                    total_missing_files += 1
                for f in corrupt:
                    print(f"        corrupt:  {f}")
                    total_corrupt_files += 1
            else:
                split_pass += 1
                total_pass += 1

        total_subjects += len(cases)
        print(f"  [{split}] {split_pass}/{len(cases)} passed")

    print("\n" + "=" * 50)
    print(f"TOTAL:  {total_pass}/{total_subjects} subjects passed")
    print(f"  Missing files : {total_missing_files}")
    print(f"  Corrupt files : {total_corrupt_files}")

    if total_pass == total_subjects:
        print("STATUS: PASS — all subjects OK")
    else:
        failed = total_subjects - total_pass
        print(f"STATUS: FAIL — {failed} subject(s) have issues")


if __name__ == "__main__":
    main()

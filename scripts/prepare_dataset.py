"""
Convert BraTS 2023 raw training split to nnUNet Dataset001_BraTS2023_GLI format.

Input layout (BraTS 2023 raw):
  <input_dir>/train/{case}/{case}-t1n.nii.gz
                          {case}-t1c.nii.gz
                          {case}-t2w.nii.gz
                          {case}-t2f.nii.gz
                          {case}-seg.nii.gz

Output layout ($nnUNet_raw/Dataset001_BraTS2023_GLI/):
  imagesTr/{case}_0000.nii.gz  <- t1n
  imagesTr/{case}_0001.nii.gz  <- t1c
  imagesTr/{case}_0002.nii.gz  <- t2w
  imagesTr/{case}_0003.nii.gz  <- t2f
  labelsTr/{case}.nii.gz       <- seg (labels 0/1/2/3, no remapping)
  dataset.json

Labels in seg files: 0=background, 1=NCR, 2=SNFH (edema), 3=ET.
Regions used: WT=1+2+3, TC=1+3, ET=3.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

MODALITY_SUFFIXES = {
    "_0000": "-t1n",
    "_0001": "-t1c",
    "_0002": "-t2w",
    "_0003": "-t2f",
}

DATASET_JSON = {
    "channel_names": {
        "0": "T1n",
        "1": "T1c",
        "2": "T2w",
        "3": "T2f",
    },
    "labels": {
        "background": 0,
        "NCR": 1,
        "SNFH": 2,
        "ET": 3,
    },
    "regions_class_order": [1, 2, 3],
    "regions": {
        "WT": [1, 2, 3],
        "TC": [1, 3],
        "ET": [3],
    },
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "name": "Dataset001_BraTS2023_GLI",
    "description": "BraTS 2023 GLI glioma segmentation dataset formatted for nnUNet v2.",
    "reference": "https://www.synapse.org/#!Synapse:syn51156910",
    "licence": "CC-BY-NC 4.0",
    "release": "2023",
    "tensorImageSize": "3D",
}


def prepare_case(case_dir: Path, images_tr: Path, labels_tr: Path) -> bool:
    """
    Symlink modality files and copy the segmentation label for one case.

    Returns True if the case was processed successfully, False if any file is missing.
    """
    case = case_dir.name
    missing = []

    for suffix, mod in MODALITY_SUFFIXES.items():
        src = case_dir / f"{case}{mod}.nii.gz"
        if not src.exists():
            missing.append(src.name)

    seg_src = case_dir / f"{case}-seg.nii.gz"
    if not seg_src.exists():
        missing.append(seg_src.name)

    if missing:
        print(f"  [warn] Skipping {case}: missing {', '.join(missing)}")
        return False

    for suffix, mod in MODALITY_SUFFIXES.items():
        src = case_dir / f"{case}{mod}.nii.gz"
        dst = images_tr / f"{case}{suffix}.nii.gz"
        if not dst.exists():
            dst.symlink_to(src.resolve())

    label_dst = labels_tr / f"{case}.nii.gz"
    if not label_dst.exists():
        label_dst.symlink_to(seg_src.resolve())

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BraTS 2023 raw data to nnUNet Dataset001_BraTS2023_GLI format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root of the BraTS 2023 raw dataset (contains train/, val/, test/ subdirs).",
    )
    parser.add_argument(
        "--nnunet_raw",
        type=Path,
        default=Path(os.environ.get("nnUNet_raw", "/pvc/nnunet/raw")),
        help="Path to nnUNet_raw directory (default: $nnUNet_raw or /pvc/nnunet/raw).",
    )
    args = parser.parse_args()

    train_dir = args.input_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training split not found: {train_dir}")

    dataset_root = args.nnunet_raw / "Dataset001_BraTS2023_GLI"
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in train_dir.iterdir() if p.is_dir())
    print(f"Found {len(cases)} case directories in {train_dir}")

    processed, skipped = 0, 0
    for case_dir in cases:
        if prepare_case(case_dir, images_tr, labels_tr):
            processed += 1
        else:
            skipped += 1

    dataset_json = DATASET_JSON.copy()
    dataset_json["numTraining"] = processed
    with open(dataset_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nSummary: {processed} cases prepared, {skipped} skipped.")
    print(f"Dataset written to: {dataset_root.resolve()}")
    print(f"Next step: nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")


if __name__ == "__main__":
    main()

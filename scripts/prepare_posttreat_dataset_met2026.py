"""
Build Dataset003_BraTS2026_MET_PostTreat — a subset of Dataset002 containing
only post-treatment training cases (i.e. cases with at least one RC voxel in
the ground truth segmentation).

Why: training on the full mixed pre+post dataset suppresses RC predictions
because ~half the cases have RC=0 everywhere. Training a dedicated model on
post-treatment cases only eliminates this negative signal.

At inference, this model is run on ALL validation cases. Its RC probability
channel replaces the full ensemble's RC channel. Pre-treatment RC zeroing
(existing postprocessing) cleans up any residual false positives.

Usage (runs inside the Docker container on the PVC):
    python /workspace/prepare_posttreat_dataset_met2026.py \
        --dataset002 /pvc/nnunet/raw/Dataset002_BraTS2026_MET \
        --nnunet_raw /pvc/nnunet/raw

Outputs:
    /pvc/nnunet/raw/Dataset003_BraTS2026_MET_PostTreat/
        imagesTr/   ← symlinks to Dataset002 imagesTr files
        labelsTr/   ← symlinks to Dataset002 labelsTr files
        imagesVal/  ← symlinks to Dataset002 imagesVal files (all 179 cases)
        dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ID = "003"
DATASET_NAME = "Dataset003_BraTS2026_MET_PostTreat"
RC_LABEL = 4


def has_rc(label_path: Path) -> bool:
    arr = np.asarray(nib.load(label_path).dataobj)
    return bool((arr == RC_LABEL).any())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset002", type=Path,
                   default=Path("/pvc/nnunet/raw/Dataset002_BraTS2026_MET"))
    p.add_argument("--nnunet_raw", type=Path,
                   default=Path("/pvc/nnunet/raw"))
    args = p.parse_args()

    src_images_tr = args.dataset002 / "imagesTr"
    src_labels_tr = args.dataset002 / "labelsTr"
    src_images_val = args.dataset002 / "imagesVal"

    dst_root = args.nnunet_raw / DATASET_NAME
    dst_images_tr = dst_root / "imagesTr"
    dst_labels_tr = dst_root / "labelsTr"
    dst_images_val = dst_root / "imagesVal"
    for d in (dst_images_tr, dst_labels_tr, dst_images_val):
        d.mkdir(parents=True, exist_ok=True)

    # --- find post-treatment cases ---
    all_labels = sorted(src_labels_tr.glob("*.nii.gz"))
    print(f"Scanning {len(all_labels)} training cases for RC voxels...")

    post_treat_cases = []
    for i, label_path in enumerate(all_labels, 1):
        if has_rc(label_path):
            post_treat_cases.append(label_path.stem.replace(".nii", ""))
        if i % 100 == 0:
            print(f"  {i}/{len(all_labels)} scanned, {len(post_treat_cases)} post-treatment so far")

    print(f"\nFound {len(post_treat_cases)} post-treatment cases out of {len(all_labels)} total")
    print(f"Pre-treatment (excluded): {len(all_labels) - len(post_treat_cases)}")

    # --- create imagesTr and labelsTr symlinks ---
    for case in post_treat_cases:
        label_src = src_labels_tr / f"{case}.nii.gz"
        label_dst = dst_labels_tr / f"{case}.nii.gz"
        if not label_dst.exists():
            label_dst.symlink_to(label_src.resolve())

        for suffix in ("_0000", "_0001", "_0002", "_0003"):
            img_src = src_images_tr / f"{case}{suffix}.nii.gz"
            img_dst = dst_images_tr / f"{case}{suffix}.nii.gz"
            if img_src.exists() and not img_dst.exists():
                img_dst.symlink_to(img_src.resolve())

    # --- symlink ALL validation cases (we want RC predictions on everything) ---
    val_cases = sorted({f.name.split("_")[0] + "_" + f.name.split("_")[1] + "_" +
                        f.name.split("_")[2] + "_" + f.name.split("_")[3]
                        for f in src_images_val.glob("*_0000.nii.gz")})
    # simpler: just symlink every file in imagesVal
    for src_file in sorted(src_images_val.glob("*.nii.gz")):
        dst_file = dst_images_val / src_file.name
        if not dst_file.exists():
            dst_file.symlink_to(src_file.resolve())

    n_val = len(list(dst_images_val.glob("*_0000.nii.gz")))

    # --- dataset.json ---
    dataset_json = {
        "channel_names": {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2f"},
        "labels": {
            "background": 0,
            "NETC": 1,
            "SNFH": 2,
            "ET": 3,
            "RC": 4,
        },
        "numTraining": len(post_treat_cases),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": (
            "BraTS 2026 MET post-treatment subset. "
            "Only cases with RC voxels in ground truth. "
            "Used to train an RC-specialist ensemble member."
        ),
        "reference": "https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097",
        "licence": "CC-BY-NC 4.0",
        "release": "2026",
        "tensorImageSize": "3D",
    }
    with open(dst_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset ready: {dst_root}")
    print(f"  imagesTr : {len(list(dst_images_tr.glob('*_0000.nii.gz')))} cases")
    print(f"  labelsTr : {len(list(dst_labels_tr.glob('*.nii.gz')))} cases")
    print(f"  imagesVal: {n_val} cases")
    print(f"\nNext step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -np 4")


if __name__ == "__main__":
    main()

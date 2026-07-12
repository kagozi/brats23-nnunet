"""
Prepare BraTS 2026 MET dataset for nnUNet (Dataset002_BraTS2026_MET).

Input zip files (place in --zip_dir):
  MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip
  MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip
  MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip

Steps performed:
  1. Extract training zip (handles nested UCSD - Training subdirectory)
  2. Extract validation zip
  3. Override 2 seg files with corrected labels
  4. Remap anomalous label 6 → 0 in BraTS-MET-01094-002
  5. Build nnUNet symlink structure + dataset.json

Output layout ($nnUNet_raw/Dataset002_BraTS2026_MET/):
  imagesTr/{case}_0000.nii.gz  ← T1n
  imagesTr/{case}_0001.nii.gz  ← T1c
  imagesTr/{case}_0002.nii.gz  ← T2w
  imagesTr/{case}_0003.nii.gz  ← T2f
  labelsTr/{case}.nii.gz       ← seg (labels 0-4)
  imagesVal/{case}_0000..0003.nii.gz
  dataset.json

Labels: 0=background, 1=NETC, 2=SNFH, 3=ET, 4=RC

Usage:
  python scripts/prepare_dataset_met2026.py \\
      --zip_dir /pvc/data/brats2026-met/zips \\
      --raw_dir /pvc/data/brats2026-met \\
      --nnunet_raw /pvc/nnunet/raw

  # Skip re-extraction if already done:
  python scripts/prepare_dataset_met2026.py ... --skip_extract
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

DATASET_ID = "002"
DATASET_NAME = "Dataset002_BraTS2026_MET"

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
        "NETC": 1,
        "SNFH": 2,
        "ET": 3,
        "RC": 4,
    },
    "numTraining": 0,
    "file_ending": ".nii.gz",
    "name": DATASET_NAME,
    "description": "BraTS 2026 Brain Metastases segmentation (pre- and post-treatment).",
    "reference": "https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097",
    "licence": "CC-BY-NC 4.0",
    "release": "2026",
    "tensorImageSize": "3D",
}

# Cases fixed by the corrected-labels zip (seg overrides already applied to these)
CORRECTED_CASES = {"BraTS-MET-01094-003", "BraTS-MET-01184-002"}

# Cases with anomalous labels NOT covered by the corrected-labels zip.
# BraTS-MET-01094-002 has 129 voxels of label 6 (unknown origin) → remap to 0.
REMAP_LABELS: Dict[str, Dict[int, int]] = {
    "BraTS-MET-01094-002": {6: 0},
}


def _nii_entry(z: zipfile.ZipFile, entry: zipfile.ZipInfo) -> Optional[Tuple[str, str]]:
    """Return (case, filename) for a .nii.gz entry, handling depth-2 and depth-3 paths."""
    parts = entry.filename.split("/")
    if not entry.filename.endswith(".nii.gz"):
        return None
    if len(parts) == 3:
        return parts[1], parts[2]
    if len(parts) == 4:
        return parts[2], parts[3]
    return None


def extract_zip(zip_path: Path, out_dir: Path, desc: str) -> None:
    """Stream-extract a training or validation zip into out_dir/case/file."""
    print(f"Extracting {desc} from {zip_path.name} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        entries = [e for e in z.infolist() if _nii_entry(z, e)]
        for i, entry in enumerate(entries, 1):
            case, fname = _nii_entry(z, entry)
            dst = out_dir / case / fname
            if dst.exists():
                continue
            dst.parent.mkdir(exist_ok=True)
            with z.open(entry) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            if i % 500 == 0:
                print(f"  {i}/{len(entries)} files extracted ...")
    print(f"  Done → {out_dir}  ({len(list(out_dir.iterdir()))} case dirs)")


def apply_corrected_labels(corrected_zip: Path, train_dir: Path) -> None:
    """Override seg files with the corrected-labels batch."""
    print(f"Applying corrected labels ...")
    with zipfile.ZipFile(corrected_zip) as z:
        for entry in z.infolist():
            if not entry.filename.endswith("-seg.nii.gz"):
                continue
            fname = Path(entry.filename).name
            case = fname.replace("-seg.nii.gz", "")
            case_dir = train_dir / case
            if not case_dir.exists():
                print(f"  [warn] Case dir missing for corrected label: {case}")
                continue
            dst = case_dir / fname
            with z.open(entry) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            print(f"  Corrected label applied: {case}")


def remap_anomalous_labels(train_dir: Path) -> None:
    """Fix cases with out-of-range label values by remapping in-place."""
    for case, remap in REMAP_LABELS.items():
        seg_path = train_dir / case / f"{case}-seg.nii.gz"
        if not seg_path.exists():
            print(f"  [warn] Seg not found for label remap: {seg_path}")
            continue
        img = nib.load(str(seg_path))
        arr = np.asarray(img.dataobj, dtype=np.int16).copy()
        for bad, good in remap.items():
            mask = arr == bad
            if mask.any():
                arr[mask] = good
                print(f"  {case}: remapped label {bad} → {good}  ({mask.sum()} voxels)")
        nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(seg_path))


def build_nnunet_structure(
    train_dir: Path,
    val_dir: Path,
    dataset_root: Path,
) -> Tuple[int, int]:
    """Create nnUNet imagesTr/labelsTr/imagesVal via symlinks."""
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_val = dataset_root / "imagesVal"
    for d in (images_tr, labels_tr, images_val):
        d.mkdir(parents=True, exist_ok=True)

    skipped = []
    n_tr = 0
    train_cases = sorted(
        p.name for p in train_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS-MET-")
    )
    for case in train_cases:
        case_dir = train_dir / case
        missing = [
            mod for mod in MODALITY_SUFFIXES.values()
            if not (case_dir / f"{case}{mod}.nii.gz").exists()
        ]
        seg_src = case_dir / f"{case}-seg.nii.gz"
        if missing or not seg_src.exists():
            skipped.append(case)
            continue
        for suffix, mod in MODALITY_SUFFIXES.items():
            dst = images_tr / f"{case}{suffix}.nii.gz"
            if not dst.exists():
                dst.symlink_to((case_dir / f"{case}{mod}.nii.gz").resolve())
        label_dst = labels_tr / f"{case}.nii.gz"
        if not label_dst.exists():
            label_dst.symlink_to(seg_src.resolve())
        n_tr += 1

    n_val = 0
    val_cases = sorted(
        p.name for p in val_dir.iterdir()
        if p.is_dir() and p.name.startswith("BraTS-MET-")
    )
    for case in val_cases:
        case_dir = val_dir / case
        for suffix, mod in MODALITY_SUFFIXES.items():
            src = case_dir / f"{case}{mod}.nii.gz"
            if not src.exists():
                continue
            dst = images_val / f"{case}{suffix}.nii.gz"
            if not dst.exists():
                dst.symlink_to(src.resolve())
        n_val += 1

    if skipped:
        print(f"  [warn] Skipped {len(skipped)} incomplete cases: {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    print(f"\nnnUNet structure ready at {dataset_root}:")
    print(f"  imagesTr : {len(list(images_tr.glob('*_0000.nii.gz')))} cases")
    print(f"  labelsTr : {len(list(labels_tr.glob('*.nii.gz')))} cases")
    print(f"  imagesVal: {len(list(images_val.glob('*_0000.nii.gz')))} cases")
    return n_tr, n_val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare BraTS 2026 MET data for nnUNet Dataset002_BraTS2026_MET."
    )
    parser.add_argument(
        "--zip_dir", type=Path, required=True,
        help="Directory containing the 3 zip files from Synapse.",
    )
    parser.add_argument(
        "--raw_dir", type=Path, required=True,
        help="Root for extracted NIfTI files (e.g. /pvc/data/brats2026-met). "
             "Subdirs training/ and validation/ will be created here.",
    )
    parser.add_argument(
        "--nnunet_raw", type=Path,
        default=Path(os.environ.get("nnUNet_raw", "/pvc/nnunet/raw")),
        help="Path to nnUNet_raw directory.",
    )
    parser.add_argument(
        "--skip_extract", action="store_true",
        help="Skip zip extraction (use if already extracted).",
    )
    args = parser.parse_args()

    train_dir = args.raw_dir / "training"
    val_dir = args.raw_dir / "validation"
    dataset_root = args.nnunet_raw / DATASET_NAME

    train_zip = args.zip_dir / "MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip"
    val_zip = args.zip_dir / "MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip"
    corrected_zip = args.zip_dir / "MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip"

    for p in (train_zip, val_zip, corrected_zip):
        if not p.exists():
            raise FileNotFoundError(f"Zip not found: {p}")

    if not args.skip_extract:
        extract_zip(train_zip, train_dir, "training data")
        extract_zip(val_zip, val_dir, "validation data")
        apply_corrected_labels(corrected_zip, train_dir)
        remap_anomalous_labels(train_dir)
    else:
        print("Skipping extraction (--skip_extract).")

    n_tr, n_val = build_nnunet_structure(train_dir, val_dir, dataset_root)

    ds_json = {**DATASET_JSON, "numTraining": n_tr}
    json_path = dataset_root / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(ds_json, f, indent=2)

    print(f"\nSummary: {n_tr} training cases, {n_val} validation cases.")
    print(f"dataset.json → {json_path}")
    print(f"\nNext step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -np 8")


if __name__ == "__main__":
    main()

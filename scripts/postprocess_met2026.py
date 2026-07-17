"""
Post-processing for BraTS 2026 MET predictions.

Rules applied in order:
1. Zero RC (label 4) for pre-treatment cases.
   Pre-treatment = the session with the lowest session ID per patient.
   RC never exists at baseline, so predicting it there is always wrong.
2. Remove small RC connected components (< --min_rc_voxels, default 50).
   Eliminates isolated false-positive RC specks in post-treatment cases.

Usage:
    python postprocess_met2026.py \
        --pred_dir /pvc/nnunet/predictions/met2026/ensemble \
        --out_dir  /pvc/nnunet/predictions/met2026/ensemble_postproc

The script writes fixed .nii.gz files to out_dir (same filenames).
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label


RC_LABEL = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="Directory with predicted .nii.gz files")
    p.add_argument("--out_dir",  required=True, help="Output directory")
    p.add_argument("--min_rc_voxels", type=int, default=50,
                   help="Remove RC connected components smaller than this (default: 50)")
    return p.parse_args()


def parse_case_id(filename: str):
    """
    BraTS-MET-{PatientID}-{SessionID}.nii.gz
    Returns (patient_id, session_id) as strings, or None if pattern doesn't match.
    """
    stem = filename.replace(".nii.gz", "")
    parts = stem.split("-")
    if len(parts) >= 4 and parts[0] == "BraTS" and parts[1] == "MET":
        patient_id = parts[2]
        session_id = parts[3]
        return patient_id, session_id
    return None, None


def find_pretreatment_cases(pred_files: list[Path]) -> set[str]:
    """
    For each patient, the session with the numerically lowest session ID
    is considered pre-treatment. Returns a set of filenames (stems) to zero RC on.
    """
    patient_sessions: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for f in pred_files:
        pid, sid = parse_case_id(f.name)
        if pid is not None:
            patient_sessions[pid].append((sid, f))

    pre_treatment: set[str] = set()
    for pid, sessions in patient_sessions.items():
        if len(sessions) > 1:
            # Only zero RC when we can confirm it's the baseline session
            earliest_sid = min(s for s, _ in sessions)
            for sid, f in sessions:
                if sid == earliest_sid:
                    pre_treatment.add(f.name)
        # Single-session patients: can't confirm baseline vs post-treatment,
        # leave RC unchanged to avoid harming single-session cases

    return pre_treatment


def remove_small_rc_components(data: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    """Zero out RC connected components smaller than min_voxels. Returns (data, n_removed)."""
    rc_mask = data == RC_LABEL
    if not rc_mask.any():
        return data, 0
    labeled, n_components = cc_label(rc_mask)
    removed = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if comp_mask.sum() < min_voxels:
            data[comp_mask] = 0
            removed += 1
    return data, removed


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"No .nii.gz files found in {pred_dir}")
        return

    pre_treatment = find_pretreatment_cases(pred_files)
    print(f"Found {len(pred_files)} predictions, {len(pre_treatment)} identified as pre-treatment")
    print(f"RC small-component threshold: {args.min_rc_voxels} voxels")

    zeroed = 0
    total_cc_removed = 0
    for f in pred_files:
        out_path = out_dir / f.name
        img = nib.load(f)
        data = np.asarray(img.dataobj).copy()
        modified = False

        # Rule 1: zero all RC in pre-treatment cases
        if f.name in pre_treatment:
            rc_voxels = (data == RC_LABEL).sum()
            data[data == RC_LABEL] = 0
            print(f"  [pre-tx RC zeroed] {f.name}  ({rc_voxels} voxels)")
            zeroed += 1
            modified = True

        # Rule 2: remove small RC components (applies to all cases)
        data, n_removed = remove_small_rc_components(data, args.min_rc_voxels)
        if n_removed:
            print(f"  [RC CC filter]     {f.name}  ({n_removed} small components removed)")
            total_cc_removed += n_removed
            modified = True

        if modified:
            nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
        else:
            shutil.copy2(f, out_path)

    print(f"\nDone.")
    print(f"  RC zeroed (pre-treatment): {zeroed} cases")
    print(f"  RC small components removed: {total_cc_removed} total across all cases")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()

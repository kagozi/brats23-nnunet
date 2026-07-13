"""
Post-processing for BraTS 2026 MET predictions.

Key rule: Zero RC (label 4) for pre-treatment cases.
Pre-treatment = the session with the lowest session ID per patient.
After surgery / SRS, RC (Residual Cavity) can appear; it never exists
at baseline, so predicting it there is always wrong.

Usage:
    python postprocess_met2026.py \
        --pred_dir /pvc/nnunet/predictions/met2026/ensemble \
        --out_dir  /pvc/nnunet/predictions/met2026/ensemble_postproc

The script writes fixed .nii.gz files to out_dir (same filenames).
Cases that are not pre-treatment are copied unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np


RC_LABEL = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True, help="Directory with predicted .nii.gz files")
    p.add_argument("--out_dir",  required=True, help="Output directory")
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

    zeroed = 0
    for f in pred_files:
        out_path = out_dir / f.name
        if f.name in pre_treatment:
            img = nib.load(f)
            data = np.asarray(img.dataobj).copy()
            rc_voxels = (data == RC_LABEL).sum()
            data[data == RC_LABEL] = 0
            nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
            print(f"  [RC zeroed] {f.name}  ({rc_voxels} voxels removed)")
            zeroed += 1
        else:
            shutil.copy2(f, out_path)

    print(f"\nDone. RC zeroed in {zeroed} pre-treatment cases. Output: {out_dir}")


if __name__ == "__main__":
    main()

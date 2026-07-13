"""
Evaluation script for BraTS 2026 MET predictions.

Computes per-label and mean Dice (DSC) and Normalized Surface Distance (NSD)
at 1mm tolerance for labels: NETC=1, SNFH=2, ET=3, RC=4.

Usage:
    python evaluate_met2026.py \
        --pred_dir /pvc/nnunet/predictions/met2026/final \
        --gt_dir   /pvc/nnunet/raw/Dataset002_BraTS2026_MET/labelsTr \
        --out_csv  /pvc/nnunet/predictions/met2026/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
NSD_TOLERANCE_MM = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir",   required=True)
    p.add_argument("--out_csv",  default="metrics_met2026.csv")
    p.add_argument("--tolerance", type=float, default=NSD_TOLERANCE_MM,
                   help="NSD tolerance in mm (default 1.0)")
    return p.parse_args()


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty → perfect
    return float(2 * tp / denom)


def nsd(pred: np.ndarray, gt: np.ndarray, spacing: tuple, tolerance: float) -> float:
    """Normalized Surface Distance at given tolerance (mm)."""
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)

    pred_empty = pred_surf.sum() == 0
    gt_empty   = gt_surf.sum()   == 0

    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0

    # Distance from gt surface to nearest pred surface voxel (and vice versa)
    pred_dt = distance_transform_edt(~pred_surf, sampling=spacing)
    gt_dt   = distance_transform_edt(~gt_surf,   sampling=spacing)

    nsd_pred_to_gt = float((gt_dt[pred_surf]   <= tolerance).mean())
    nsd_gt_to_pred = float((pred_dt[gt_surf]   <= tolerance).mean())

    return (nsd_pred_to_gt + nsd_gt_to_pred) / 2.0


def evaluate_case(pred_path: Path, gt_path: Path, tolerance: float) -> dict:
    pred_img = nib.load(pred_path)
    gt_img   = nib.load(gt_path)

    pred = np.asarray(pred_img.dataobj).astype(np.int32)
    gt   = np.asarray(gt_img.dataobj).astype(np.int32)

    # Voxel spacing in mm (z, y, x)
    spacing = tuple(float(s) for s in pred_img.header.get_zooms()[:3])

    row: dict = {"case": pred_path.stem.replace(".nii", "")}
    dscs, nsds = [], []

    for label_id, label_name in LABELS.items():
        p_mask = pred == label_id
        g_mask = gt   == label_id

        dsc_val = dice(p_mask, g_mask)
        nsd_val = nsd(p_mask, g_mask, spacing, tolerance)

        row[f"DSC_{label_name}"] = round(dsc_val, 4)
        row[f"NSD_{label_name}"] = round(nsd_val, 4)
        dscs.append(dsc_val)
        nsds.append(nsd_val)

    row["DSC_mean"] = round(float(np.mean(dscs)), 4)
    row["NSD_mean"] = round(float(np.mean(nsds)), 4)
    return row


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)
    out_csv  = Path(args.out_csv)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"No predictions found in {pred_dir}")
        return

    rows = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"  [skip] No GT for {pred_path.name}")
            continue
        row = evaluate_case(pred_path, gt_path, args.tolerance)
        rows.append(row)
        print(
            f"  {row['case']:40s}  "
            f"DSC={row['DSC_mean']:.4f}  NSD={row['NSD_mean']:.4f}"
        )

    if not rows:
        print("No cases evaluated.")
        return

    # Summary
    label_cols = [f"{m}_{l}" for m in ("DSC", "NSD") for l in list(LABELS.values()) + ["mean"]]
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("=" * 60)
    for col in label_cols:
        vals = np.array([r[col] for r in rows if col in r])
        if len(vals):
            print(f"{col:<20} {vals.mean():>8.4f} {vals.std(ddof=1):>8.4f} {np.median(vals):>8.4f}")

    # Write CSV
    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-case metrics to {out_csv}")


if __name__ == "__main__":
    main()

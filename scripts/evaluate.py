"""
Compute WT/TC/ET Dice scores between nnUNet predictions and BraTS 2023 ground truth.

Label conventions
-----------------
BraTS 2023 GT:  0=background, 1=NCR, 2=SNFH (edema), 3=ET
Dataset137 pred: 0=background, 1=edema, 2=NCR, 3=ET  (default for this script)

Use --pred_ncr_label and --pred_et_label to override for non-Dataset137 models.

Tumor regions computed:
  WT = NCR + SNFH + ET  (whole tumour)
  TC = NCR + ET         (tumour core)
  ET = ET only          (enhancing tumour)

Output: per-subject CSV + printed mean±std summary.
NaN is used when both prediction and GT are empty for a region (avoids false penalisation).
"""

import argparse
import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Returns float('nan') when both masks are empty (no structure present in either).
    Returns 0.0 when only one mask is empty.
    """
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    if pred_sum == 0 and gt_sum == 0:
        return float("nan")
    intersection = (pred_mask & gt_mask).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def load_volume(path: Path) -> np.ndarray:
    """Load a NIfTI file and return its data as a uint8 numpy array."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.uint8)


def compute_region_masks(
    vol: np.ndarray,
    ncr_label: int,
    et_label: int,
    edema_label: int,
) -> dict[str, np.ndarray]:
    """
    Build binary masks for WT, TC, and ET from a label volume.

    Parameters
    ----------
    vol        : integer label array
    ncr_label  : label index for NCR (necrotic core)
    et_label   : label index for ET (enhancing tumour)
    edema_label: label index for SNFH/edema
    """
    ncr = vol == ncr_label
    et = vol == et_label
    edema = vol == edema_label
    return {
        "WT": ncr | edema | et,
        "TC": ncr | et,
        "ET": et,
    }


def evaluate_subject(
    pred_path: Path,
    gt_path: Path,
    pred_ncr_label: int,
    pred_et_label: int,
) -> dict[str, float]:
    """
    Return a dict with WT, TC, ET Dice scores for one subject.

    GT convention:   NCR=1, edema=2, ET=3 (BraTS 2023)
    Pred convention: NCR=pred_ncr_label, edema=1, ET=pred_et_label (Dataset137 default)
    """
    pred = load_volume(pred_path)
    gt = load_volume(gt_path)

    # Infer edema label in pred as whichever of {1,2,3} is not NCR or ET
    all_labels = {1, 2, 3}
    pred_edema_label = (all_labels - {pred_ncr_label, pred_et_label}).pop()

    pred_masks = compute_region_masks(pred, pred_ncr_label, pred_et_label, pred_edema_label)
    gt_masks = compute_region_masks(gt, ncr_label=1, et_label=3, edema_label=2)

    return {region: dice(pred_masks[region], gt_masks[region]) for region in ("WT", "TC", "ET")}


def nanmean(values: list[float]) -> float:
    """Mean of a list, ignoring NaN entries."""
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def nanstd(values: list[float]) -> float:
    """Std of a list, ignoring NaN entries."""
    finite = [v for v in values if not math.isnan(v)]
    return float(np.std(finite, ddof=0)) if len(finite) > 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nnUNet predictions against BraTS 2023 ground truth."
    )
    parser.add_argument(
        "--pred_dir",
        type=Path,
        required=True,
        help="Directory containing nnUNet prediction NIfTI files ({case}.nii.gz).",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        required=True,
        help="Directory containing BraTS 2023 ground-truth seg files ({case}-seg.nii.gz).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/pvc/nnunet/eval"),
        help="Directory to write per-subject CSV (default: /pvc/nnunet/eval).",
    )
    parser.add_argument(
        "--pred_ncr_label",
        type=int,
        default=2,
        help="Label index for NCR in prediction files (default: 2 for Dataset137).",
    )
    parser.add_argument(
        "--pred_et_label",
        type=int,
        default=3,
        help="Label index for ET in prediction files (default: 3 for Dataset137).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(args.pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {args.pred_dir}")

    rows: list[dict] = []
    missing_gt = 0

    for pred_path in pred_files:
        case = pred_path.name.replace(".nii.gz", "")

        # GT files follow BraTS convention: {case}-seg.nii.gz
        gt_path = args.gt_dir / f"{case}-seg.nii.gz"
        if not gt_path.exists():
            # Fallback: gt stored as {case}.nii.gz (e.g. already extracted)
            gt_path = args.gt_dir / f"{case}.nii.gz"
        if not gt_path.exists():
            print(f"  [warn] GT not found for {case}, skipping.")
            missing_gt += 1
            continue

        scores = evaluate_subject(pred_path, gt_path, args.pred_ncr_label, args.pred_et_label)
        row = {"subject": case, "WT": scores["WT"], "TC": scores["TC"], "ET": scores["ET"]}
        rows.append(row)

    csv_path = args.output_dir / "dice_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "WT", "TC", "ET"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject": row["subject"],
                    "WT": f"{row['WT']:.4f}" if not math.isnan(row["WT"]) else "nan",
                    "TC": f"{row['TC']:.4f}" if not math.isnan(row["TC"]) else "nan",
                    "ET": f"{row['ET']:.4f}" if not math.isnan(row["ET"]) else "nan",
                }
            )

    print(f"\nResults written to: {csv_path}")
    print(f"Evaluated: {len(rows)} subjects  |  GT missing: {missing_gt}\n")

    for region in ("WT", "TC", "ET"):
        vals = [r[region] for r in rows]
        m = nanmean(vals)
        s = nanstd(vals)
        n_nan = sum(math.isnan(v) for v in vals)
        print(f"  {region}:  {m:.4f} ± {s:.4f}  (NaN excluded: {n_nan})")


if __name__ == "__main__":
    main()

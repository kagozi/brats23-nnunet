# BraTS 2023 + nnUNet

This repository provides a self-contained pipeline for running a pretrained nnUNet model (Dataset137, trained on BraTS-2021) against the BraTS 2023 brain tumour dataset, evaluating segmentation performance.

---

## 1. Overview

**Goal**: Segment brain tumours (WT / TC / ET) in multi-modal MRI without retraining, using a high-quality pretrained nnUNet, and lay the groundwork for a Prototype-Guided Explainable Segmentation system by exposing the encoder's intermediate feature maps.

**What this repo does**:
- Downloads the Dataset137 checkpoint (BraTS-2021 pretrained, fold 5).
- Converts BraTS 2023 raw data into nnUNet-compatible format.
- Runs 3D full-resolution nnUNet inference.
- Evaluates WT / TC / ET Dice against BraTS 2023 ground truth, handling the label convention mismatch between Dataset137 and BraTS 2023.
- Provides Kubernetes job manifests for Nautilus HyperCluster.

---

## 2. Dataset

### BraTS 2023 GLI (glioma) + MET (metastasis)
**Modalities per subject** (4-channel 3D NIfTI, 1mm isotropic, skull-stripped):

| Channel | File suffix | Description |
|---------|-------------|-------------|
| 0 | `-t1n.nii.gz` | T1 native |
| 1 | `-t1c.nii.gz` | T1 contrast-enhanced |
| 2 | `-t2w.nii.gz` | T2 weighted |
| 3 | `-t2f.nii.gz` | T2 FLAIR |

**Raw folder structure**:
```
brats23/
  train/   (1044 subjects)
    BraTS-GLI-00000-000/
      BraTS-GLI-00000-000-t1n.nii.gz
      BraTS-GLI-00000-000-t1c.nii.gz
      BraTS-GLI-00000-000-t2w.nii.gz
      BraTS-GLI-00000-000-t2f.nii.gz
      BraTS-GLI-00000-000-seg.nii.gz
  val/     (148 subjects)
  test/    (297 subjects, no seg files — blind challenge)
```

**Ground-truth label convention (BraTS 2023)**:

| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | NCR — necrotic core |
| 2 | SNFH — surrounding non-enhancing FLAIR hyperintensity (edema) |
| 3 | ET — enhancing tumour |

**Tumour regions** (used in all evaluations):

| Region | Definition | Meaning |
|--------|-----------|---------|
| WT | labels 1 + 2 + 3 | Whole Tumour |
| TC | labels 1 + 3 | Tumour Core |
| ET | label 3 | Enhancing Tumour |

---

## 3. Pretrained Model — Dataset137

**Source**: nnUNet Dataset137, trained on BraTS-2021 data, fold 5.

**Architecture**: PlainConvUNet, 3D full-resolution
- 88.6M parameters
- 6 encoder stages, feature widths: 32 → 64 → 128 → 256 → 320 → 320
- Input: 4-channel 3D volume, 128×128×128 patch size
- Installed via `pip install nnunetv2`

**Checkpoint files** (downloaded by `download_model.py`):

| File | Google Drive ID |
|------|----------------|
| `checkpoint_final.pth` | `1n9dqT114udr9Qq8iYEKsJK347iHg9N88` |
| `dataset.json` | `1A_suxQwElucF3w1HEYg3wMo6dG9OxBHo` |
| `plans.json` | `1U2b0BTNi8zrJACReoi_W08Fe-wM394wI` |

**Output label convention (Dataset137)**:

| Label | Structure |
|-------|-----------|
| 0 | Background |
| 1 | Edema (SNFH) |
| 2 | NCR |
| 3 | ET |

This differs from BraTS 2023 GT (NCR=1, edema=2 vs. NCR=2, edema=1). `evaluate.py` handles this automatically via `--pred_ncr_label 2 --pred_et_label 3` (defaults).

---

## 4. Prerequisites

**Python**: 3.9+  
**CUDA**: 11.x or 12.x (for GPU inference/training)

```bash
pip install -r requirements.txt
```

**nnUNet environment variables** (required by nnUNetv2):
```bash
export nnUNet_raw=/path/to/nnunet/raw
export nnUNet_preprocessed=/path/to/nnunet/preprocessed
export nnUNet_results=/path/to/nnunet/results
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

---

## 5. Quick Start

Four commands from raw data to Dice scores:

```bash
# 1. Download pretrained checkpoint
python scripts/download_model.py --output_dir $nnUNet_results/Dataset137_BraTS2021

# 2. Reformat test data for inference (uses symlinks, no copy)
python scripts/reformat_inputs.py \
  --data_dir /data/brats23 \
  --output_dir /tmp/imagesTs \
  --split test

# 3. Run inference
nnUNetv2_predict \
  -i /tmp/imagesTs \
  -o /tmp/predictions \
  -d 137 -c 3d_fullres -f 5

# 4. Evaluate (val split has ground truth; test does not)
python scripts/evaluate.py \
  --pred_dir /tmp/predictions \
  --gt_dir /data/brats23/val \
  --output_dir /tmp/eval
```

---

## 6. Detailed Usage

### `download_model.py`

Downloads Dataset137 checkpoint files from Google Drive.

```
python download_model.py [--output_dir OUTPUT_DIR]

Arguments:
  --output_dir   Root output directory (default: ./models/Dataset137_BraTS2021)
```

Creates:
```
<output_dir>/
  nnUNetTrainer__nnUNetPlans__3d_fullres/
    fold_5/
      checkpoint_final.pth
      plans.json
      dataset.json
  dataset.json
```

Skips files that already exist. Prints file sizes after download.

---

### `check_integrity.py`

Validates that every subject directory has all 5 expected files and that each `.nii.gz` loads without error.

```
python scripts/check_integrity.py --data_dir DATA_DIR

Arguments:
  --data_dir   Root BraTS 2023 directory (must contain train/, val/, and/or test/).
```

Run this before any other script to catch download corruption early.

---

### `reformat_inputs.py`

Creates nnUNet-compatible symlinks for inference without copying data (~27 GB per split).

```
python scripts/reformat_inputs.py \
  --data_dir DATA_DIR \
  --output_dir OUTPUT_DIR \
  --split {train,val,test}

Arguments:
  --data_dir    Root BraTS 2023 raw directory.
  --output_dir  Output directory for nnUNet-style files (default: /pvc/nnunet/imagesTs).
  --split       Which split to reformat (default: test).
```

Output: `{case}_0000.nii.gz` through `{case}_0003.nii.gz` symlinked into `OUTPUT_DIR`.

---

### `prepare_dataset.py`

Converts the BraTS 2023 training split to nnUNet Dataset001_BraTS2023_GLI format for training from scratch. Also generates `dataset.json` with WT/TC/ET region definitions.

```
python scripts/prepare_dataset.py \
  --input_dir INPUT_DIR \
  --nnunet_raw NNUNET_RAW

Arguments:
  --input_dir    Root BraTS 2023 raw directory.
  --nnunet_raw   nnUNet_raw path (default: $nnUNet_raw or /pvc/nnunet/raw).
```

After running, preprocess with:
```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

---

### `evaluate.py`

Computes per-subject and aggregate WT / TC / ET Dice scores.

```
python scripts/evaluate.py \
  --pred_dir PRED_DIR \
  --gt_dir GT_DIR \
  --output_dir OUTPUT_DIR \
  [--pred_ncr_label 2] \
  [--pred_et_label 3]

Arguments:
  --pred_dir        Directory with prediction .nii.gz files ({case}.nii.gz).
  --gt_dir          Directory with BraTS 2023 GT files ({case}-seg.nii.gz or split dir).
  --output_dir      Where to write dice_scores.csv (default: /pvc/nnunet/eval).
  --pred_ncr_label  Label index for NCR in predictions (default: 2 — Dataset137).
  --pred_et_label   Label index for ET in predictions (default: 3 — Dataset137).
```

When both prediction and GT are empty for a region, Dice is set to `NaN` (not penalised).  
Output: `dice_scores.csv` with columns `subject, WT, TC, ET`, plus printed `mean ± std`.

---

## 7. Training from Scratch

Use this when you want to fine-tune or train a new model on BraTS 2023 instead of using the pretrained Dataset137 weights.

```bash
# Step 1 — Prepare the dataset
python scripts/prepare_dataset.py \
  --input_dir /data/brats23 \
  --nnunet_raw $nnUNet_raw

# Step 2 — Plan and preprocess
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

# Step 3 — Train (one fold; repeat for folds 1-4 for full cross-validation)
nnUNetv2_train 001 3d_fullres 0

# Step 4 — Predict with your trained model
nnUNetv2_predict \
  -i /tmp/imagesTs \
  -o /tmp/predictions_trained \
  -d 001 -c 3d_fullres -f 0
```

Note: When evaluating a model trained on BraTS 2023 GT labels (NCR=1, edema=2, ET=3), pass `--pred_ncr_label 1 --pred_et_label 3` to `evaluate.py`.

---

## 8. Label Convention

The BraTS 2023 ground-truth labels and Dataset137 prediction labels use different orderings for NCR and edema. ET=3 in both.

| Label index | BraTS 2023 GT | Dataset137 predictions |
|-------------|--------------|----------------------|
| 0 | Background | Background |
| 1 | NCR (necrotic core) | Edema (SNFH) |
| 2 | SNFH (edema) | NCR (necrotic core) |
| 3 | ET (enhancing tumour) | ET (enhancing tumour) |

**Why the mismatch?** Dataset137 was trained on BraTS-2021 data where the label ordering differed from BraTS 2023. Since WT, TC, ET are computed from combinations of labels, the Dice scores for WT and ET are unaffected; only TC (which includes NCR) requires knowing the correct NCR label index.

`evaluate.py` accepts `--pred_ncr_label` (default 2) and `--pred_et_label` (default 3) so you can override this for any model.

---

## 9. Architecture recap

 PlainConvUNet encoder stages:

| Stage | Feature width | Spatial scale |
|-------|-------------|--------------|
| 0 | 32 | 128³ (full res) |
| 1 | 64 | 64³ |
| 2 | 128 | 32³ |
| 3 | 256 | 16³ |
| 4 | 320 | 8³ |
| 5 | 320 | 4³ (bottleneck) |

**How to extract encoder features**:

Load the checkpoint and hook into the encoder blocks:

```python
import torch
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

model = get_network_from_plans(...)  # instantiate from plans.json
checkpoint = torch.load("checkpoint_final.pth", map_location="cpu")
model.load_state_dict(checkpoint["network_weights"])
model.eval()

# Register a forward hook on encoder stage 4 (320-channel bottleneck-adjacent)
features = {}
def hook(module, input, output):
    features["enc4"] = output.detach()

model.encoder.stages[4].register_forward_hook(hook)
```

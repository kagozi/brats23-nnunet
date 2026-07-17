# =============================================================================
# BraTS 2023 — nnU-Net segmentation + prototype-guided explainability
#
# Build:
#   docker build -t ghcr.io/kagozi/brats23-nnunet:latest .
#
# Run (example):
#   docker run --gpus all -v /pvc:/pvc ghcr.io/kagozi/brats23-nnunet:latest
# =============================================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Core scripts (referenced as /workspace/<script>.py in all K8s jobs)
COPY scripts/ /workspace/

# Register MedNextTrainerL with nnUNetv2 so `nnUNetv2_train -tr MedNextTrainerL` works
RUN cp /workspace/MedNextTrainerL.py \
       /opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/network_architecture/MedNextTrainerL.py

# Prototype experiment code
COPY prototype_nnunet_evidence/ /workspace/prototype_nnunet_evidence/

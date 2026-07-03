
import os
import sys
from pathlib import Path

import torch

sys.path.append("/kaggle/working/scripts")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from prototype_module_3d import PrototypeFusionBlock3D


def main():
    os.environ["nnUNet_raw"] = "/kaggle/working/nnunet/raw"
    os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnunet/preprocessed"
    os.environ["nnUNet_results"] = "/kaggle/working/nnunet/results"

    input_dir = "/kaggle/working/nnunet/imagesTs/val"
    output_dir = "/kaggle/working/nnunet/predictions/val/dataset137_proto"

    model_folder = (
        "/kaggle/working/nnunet/results/"
        "Dataset137_BraTS2021/"
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(5,),
        checkpoint_name="checkpoint_final.pth",
    )

    model = predictor.network.to(device)
    model.eval()

    proto_block = PrototypeFusionBlock3D(
        feature_dim=320,
        num_classes=3,
        prototypes_per_class=8,
        num_heads=4,
    ).to(device)

    proto_block.eval()

    call_counter = {"count": 0}

    def prototype_hook(module, inputs, output):
        proto_out = proto_block(output)

        if call_counter["count"] == 0:
            print("\nPrototype hook active at encoder_stage_5")
            print("Original bottleneck:", tuple(output.shape))
            print("Fused bottleneck:", tuple(proto_out["fused_features"].shape))
            print("Proto similarity:", tuple(proto_out["proto_similarity"].shape))
            print("Class evidence:", tuple(proto_out["class_evidence"].shape))
            print("Top proto indices:", tuple(proto_out["top_proto_indices"].shape))
            print("Gate:", tuple(proto_out["gate"].shape))

        call_counter["count"] += 1

        return proto_out["fused_features"]

    handle = model.encoder.stages[5].register_forward_hook(prototype_hook)

    print("\nRunning nnU-Net prediction WITH prototype bottleneck hook...")
    print("Input:", input_dir)
    print("Output:", output_dir)

    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    handle.remove()

    print("\nDone.")
    print("Prototype hook was called", call_counter["count"], "times.")
    print("Saved predictions to:", output_dir)


if __name__ == "__main__":
    main()

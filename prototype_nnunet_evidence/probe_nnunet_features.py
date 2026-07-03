
import os
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    os.environ["nnUNet_raw"] = "/kaggle/working/nnunet/raw"
    os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnunet/preprocessed"
    os.environ["nnUNet_results"] = "/kaggle/working/nnunet/results"

    model_folder = (
        "/kaggle/working/nnunet/results/"
        "Dataset137_BraTS2021/"
        "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(5,),
        checkpoint_name="checkpoint_final.pth",
    )

    model = predictor.network
    model.eval()

    print("\nModel loaded successfully.")
    print("Model type:", type(model))

    print("\nEncoder:")
    print(model.encoder)

    features = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                features[name] = tuple(output.shape)
            elif isinstance(output, (list, tuple)):
                features[name] = [tuple(o.shape) for o in output if isinstance(o, torch.Tensor)]
            else:
                features[name] = str(type(output))
        return hook

    for i, stage in enumerate(model.encoder.stages):
        stage.register_forward_hook(make_hook(f"encoder_stage_{i}"))

    x = torch.randn(1, 4, 128, 128, 128).to(next(model.parameters()).device)

    print("\nRunning dummy forward pass...")
    with torch.no_grad():
        y = model(x)

    print("\nOutput shape:")
    if isinstance(y, torch.Tensor):
        print(tuple(y.shape))
    elif isinstance(y, (list, tuple)):
        print([tuple(o.shape) for o in y if isinstance(o, torch.Tensor)])
    else:
        print(type(y))

    print("\nCaptured encoder feature shapes:")
    for k, v in features.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()

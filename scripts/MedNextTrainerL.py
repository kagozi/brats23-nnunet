import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MedNextTrainerL(nnUNetTrainer):
    """
    nnUNetv2-compatible trainer for MedNeXt-L (kernel_size=3, do_res=True).
    Uses AdamW + poly LR, matching the original MedNeXt paper setup.
    Requires nnunet_mednext to be installed (already in the Docker image).
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda'), **kwargs):
        # Base class signature is (plans, configuration, fold, dataset_json, device).
        # run_training.py also passes unpack_dataset=; absorb it in **kwargs, don't forward it.
        super().__init__(plans=plans, configuration=configuration, fold=fold,
                         dataset_json=dataset_json, device=device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        # MedNeXt v1 doesn't expose the decoder.deep_supervision interface that
        # nnUNetv2 expects, so disable DS after super().__init__ sets it True.
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=False):
        from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import MedNeXt
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        pass  # MedNeXt DS not compatible with nnUNetv2 decoder interface

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.num_epochs, power=0.9
        )
        return optimizer, lr_scheduler

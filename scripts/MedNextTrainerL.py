import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MedNextTrainerL(nnUNetTrainer):
    """
    nnUNetv2-compatible trainer for MedNeXt-L (kernel_size=3, do_res=True).
    Uses AdamW + poly LR, matching the original MedNeXt paper setup.
    Requires nnunet_mednext to be installed (already in the Docker image).
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import MedNeXt
        return MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        )

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

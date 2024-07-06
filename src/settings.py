from argparse import ArgumentParser, BooleanOptionalAction


class DefaultArgParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--device", type=str, default="cpu")
        self.add_argument("--task", type=str)
        self.add_argument("--kernel", type=str, default=None)
        self.add_argument(
            "--physics_true_adjoint", action=BooleanOptionalAction, default=False
        )
        self.add_argument("--sr_factor", type=int, default=None)
        self.add_argument("--noise_level", type=int, default=5)
        self.add_argument("--dataset", type=str, default="div2k")
        self.add_argument(
            "--GroundTruthDataset__datasets_dir", type=str, default="./datasets"
        )
        self.add_argument(
            "--GroundTruthDataset__download",
            action=BooleanOptionalAction,
            default=False,
        )
        group = self.add_mutually_exclusive_group()
        group.add_argument("--GroundTruthDataset__size", type=int, default=256)
        group.add_argument(
            "--GroundTruthDataset__no_resize",
            action="store_const",
            dest="GroundTruthDataset__size",
            const=None,
        )
        # NOTE: This should be set to true!
        self.add_argument(
            "--Dataset__unique_seeds", action=BooleanOptionalAction, default=False
        )
        # NOTE: This should ideally be in the training script but it is easier
        # to keep it right here as the preparation of training pairs is
        # involved in the class Dataset (which itself should ideally be
        # elsewhere)
        self.add_argument("--PrepareTrainingPairs__crop_size", type=int, default=48)
        self.add_argument(
            "--PrepareTrainingPairs__crop_location", type=str, default="random"
        )
        self.add_argument("--model_kind", type=str, default="swinir")
        self.add_argument("--unet_residual", action=BooleanOptionalAction, default=True)
        self.add_argument(
            "--UNet__inner_residual", action=BooleanOptionalAction, default=True
        )
        self.add_argument("--unet_num_conv_blocks", type=int, default=5)
        self.add_argument("--SingleImageDataset__image_path", type=str, default=None)
        self.add_argument(
            "--SingleImageDataset__duplicates_count", type=int, default=800
        )
        self.add_argument("--data_parallel_devices", type=str, default=None)

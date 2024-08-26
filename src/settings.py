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
            "--download",
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
        self.add_argument(
            "--SyntheticDataset__unique_seeds",
            action=BooleanOptionalAction,
            default=True,
        )
        # NOTE: This should ideally be in the training script but it is easier
        # to keep it right here as the preparation of training pairs is
        # involved in the class Dataset (which itself should ideally be
        # elsewhere)
        # NOTE: This is pretty much meant to be the size of ground truth
        # images, i.e. it crop squares of the largest size. For this reason, it
        # is awkward to have to specify the size here as well.
        self.add_argument("--PrepareTrainingPairs__crop_size", type=int, default=256)
        self.add_argument(
            "--PrepareTrainingPairs__crop_location", type=str, default="random"
        )
        self.add_argument("--model_kind", type=str, default="Proposed")
        self.add_argument("--ProposedModel__architecture", type=str, default="Transformer")
        self.add_argument("--ConvolutionalModel__residual", action=BooleanOptionalAction, default=True)
        self.add_argument(
            "--ConvolutionalModel__inner_residual", action=BooleanOptionalAction, default=True
        )
        self.add_argument(
            "--ConvolutionalModel__inout_convs", action=BooleanOptionalAction, default=True
        )
        self.add_argument("--ConvolutionalModel__hidden_channels", type=int, default=32)
        self.add_argument("--ConvolutionalModel__scales", type=int, default=5)
        self.add_argument("--ConvolutionalModel__num_conv_blocks", type=int, default=1)
        self.add_argument("--SingleImageDataset__image_path", type=str, default=None)
        self.add_argument(
            "--SingleImageDataset__duplicates_count", type=int, default=800
        )
        self.add_argument("--data_parallel_devices", type=str, default=None)
        self.add_argument("--physics_v2", action=BooleanOptionalAction, default=True)

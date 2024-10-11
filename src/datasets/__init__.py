from torch.nn import Module
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import functional as TF

from crop import CropPair
from noise2inverse import Noise2InverseTransform
from .ground_truth import GroundTruthDataset
from .synthetic_dataset import SyntheticDataset
from .single_image import SingleImageDataset


# NOTE: Getting small random crops should be optional and it should
# be possible to use big crops instead, e.g. to make images
# square-shaped which would enable stacking in the batch dimension.
class PrepareTrainingPairs(Module):
    def __init__(self, physics, crop_size, crop_location):
        super().__init__()
        self.physics = physics
        self.crop_size = crop_size
        self.crop_location = crop_location
        from os import environ
        if "HOMOGENEOUS_SWINIR" in environ:
            if "_once452" not in globals():
                print("\nCropping training patches of the same size\n")
                globals()["_once452"] = True
            print("\nSetting a training crop size of 48 pixels\n")
            self.crop_size = 48

    def forward(self, x, y):
        # NOTE: It'd be great if physics contained its own downsampling ratio
        # even for a blur operator.

        if self.physics.task == "sr":
            xy_size_ratio = self.physics.rate
            from os import environ
            if "HOMOGENEOUS_SWINIR" in environ:
                if "_once451" not in globals():
                    print("\nCropping training patches of the same size\n")
                    globals()["_once451"] = True
                xy_size_ratio = 1

        else:
            xy_size_ratio = 1

        T_crop = CropPair(location=self.crop_location, size=self.crop_size)
        return T_crop(x, y, xy_size_ratio=xy_size_ratio)


class TrainingDataset(BaseDataset):
    def __init__(
        self,
        synthetic_dataset,
        physics,
        css,
        noise2inverse,
        prepare_training_pairs,
        _HOTFIX,
    ):
        super().__init__()
        self.synthetic_dataset = synthetic_dataset
        self.physics = physics
        self.css = css
        self.noise2inverse = noise2inverse
        self.prepare_training_pairs = prepare_training_pairs
        self.important_unnamed_flag = _HOTFIX

    def __getitem__(self, index):
        x, y = self.synthetic_dataset[index]

        if self.css:
            physics_manager = getattr(self.physics, "__manager")
            y = y.unsqueeze(0)
            z = physics_manager.randomly_degrade(y, seed=None)
            z = z.squeeze(0)
            y = y.squeeze(0)
            x, y = y, z

        if self.important_unnamed_flag:
            from os import environ
            if "HOMOGENEOUS_SWINIR" in environ:
                x, y = self.prepare_training_pairs(x, y)
                return x, y

            T_crop = CropPair(location="random", size=48)
            return T_crop(x, y, xy_size_ratio=self.physics.rate)
        else:
            # NOTE: This should ideally either be done in the model, or not at
            # all.
            x, y = self.prepare_training_pairs(x, y)
            return x, y

    def __len__(self):
        return len(self.synthetic_dataset)


class TestDataset(BaseDataset):
    def __init__(
        self,
        synthetic_dataset,
        noise2inverse,
        physics,
    ):
        super().__init__()
        self.synthetic_dataset = synthetic_dataset
        self.noise2inverse = noise2inverse
        self.physics = physics

    def __getitem__(self, index):
        x, y = self.synthetic_dataset[index]

        # NOTE: This should ideally be removed.
        if self.noise2inverse:
            # bug fix: make y have even height and width
            if self.physics.task == "deblurring":
                w = 2 * (y.shape[1] // 2)
                h = 2 * (y.shape[2] // 2)
                y = y[:, :w, :h]

        # NOTE: This should ideally be removed.
        # crop x to make its dimensions be a multiple of u's dimensions
        if x.shape != y.shape:
            h, w = y.shape[1], y.shape[2]
            if self.physics.task == "sr":
                f = self.physics.rate
            else:
                f = 1
            x = TF.crop(x, top=0, left=0, height=h * f, width=w * f)

        return x, y

    def __len__(self):
        return len(self.synthetic_dataset)


class Dataset(BaseDataset):
    def __init__(
        self,
        blueprint,
        purpose,
        physics,
        css,
        noise2inverse,
        device,
        _HOTFIX,
    ):
        super().__init__()

        synthetic_dataset = SyntheticDataset(
            blueprint=blueprint,
            device=device,
            physics=physics,
            **blueprint[SyntheticDataset.__name__],
        )

        if purpose == "train":
            prepare_training_pairs = PrepareTrainingPairs(
                physics=physics,
                **blueprint[PrepareTrainingPairs.__name__],
            )

            self.dataset = TrainingDataset(
                synthetic_dataset=synthetic_dataset,
                physics=physics,
                css=css,
                noise2inverse=noise2inverse,
                prepare_training_pairs=prepare_training_pairs,
                _HOTFIX=_HOTFIX
            )
        elif purpose == "test":
            self.dataset = TestDataset(
                synthetic_dataset=synthetic_dataset,
                noise2inverse=noise2inverse,
                physics=physics,
            )
        else:
            raise ValueError(f"Unknown purpose: {purpose}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_dataset(args, purpose, physics, device, _HOTFIX):
    if purpose == "test":
        noise2inverse = args.noise2inverse
        css = False
    elif purpose == "train":
        noise2inverse = args.method == "noise2inverse"
        css = args.method == "css"
    else:
        raise ValueError(f"Unknown purpose: {purpose}")

    blueprint = {}

    blueprint[GroundTruthDataset.__name__] = {
        # NOTE: This argument should be named according to the class
        # GroundTruthDataset but happens to be used (wrongly) elsewhere and
        # this must be dealt with first
        "dataset_name": args.dataset,
        "datasets_dir": args.GroundTruthDataset__datasets_dir,
        "download": args.GroundTruthDataset__download,
        "size": args.GroundTruthDataset__size,
        "split": args.GroundTruthDataset__split,
        "memoize_gt": args.memoize_gt,
    }

    blueprint[PrepareTrainingPairs.__name__] = {
        "crop_size": args.PrepareTrainingPairs__crop_size,
        "crop_location": args.PrepareTrainingPairs__crop_location,
    }

    blueprint[SingleImageDataset.__name__] = {
        "image_path": args.SingleImageDataset__image_path,
        "duplicates_count": args.SingleImageDataset__duplicates_count,
    }

    blueprint[SyntheticDataset.__name__] = {
        "unique_seeds": args.SyntheticDataset__unique_seeds,
        "deterministic_measurements": args.SyntheticDataset__deterministic_measurements,
    }

    return Dataset(
        blueprint=blueprint,
        device=device,
        physics=physics,
        purpose=purpose,
        css=css,
        noise2inverse=noise2inverse,
        _HOTFIX=_HOTFIX,
    )

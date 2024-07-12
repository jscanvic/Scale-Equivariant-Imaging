import torch
from torch.nn import Module
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import InterpolationMode, functional as TF
from functools import wraps

from noise2inverse import Noise2InverseTransform
from .crop import CropPair
from .div2k import Div2K
from .tomography import TomographyDataset
from .urban100 import Urban100
from .single_image import SingleImageDataset


class GroundTruthDataset(BaseDataset):
    def __init__(
        self,
        blueprint,
        datasets_dir,
        dataset_name,
        split,
        download,
        size,
        memoize_gt,
    ):
        super().__init__()
        self.size = size
        self.memoize_gt = memoize_gt

        if dataset_name == "div2k":
            self.dataset = Div2K(split, datasets_dir, download=download)
        elif dataset_name == "urban100":
            self.dataset = Urban100(split, datasets_dir, download=download)
        elif dataset_name == "ct":
            self.dataset = TomographyDataset(split, datasets_dir, download=download)
        elif dataset_name == "single_image":
            self.dataset = SingleImageDataset(**blueprint[SingleImageDataset.__name__])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_unique_id(self, index):
        if hasattr(self.dataset, "get_unique_id"):
            id = self.dataset.get_unique_id(index)
        else:
            id = index
        return id

    @staticmethod
    def memoize_load_image(f):
        cache = {}

        @wraps(f)
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self.memoize_gt:
                x = f(*args, **kwargs)
            else:
                key = (args, frozenset(kwargs.items()))
                if key not in cache:
                    x = f(*args, **kwargs)
                    device = x.device
                    # NOTE: This should be unnecessary.
                    x = x.to("cpu")
                    cache[key] = (device, x)
                device, x = cache[key]
                x = x.to(device)
            return x

        return wrapper

    @memoize_load_image
    def __getitem__(self, index):
        x = self.dataset[index]
        if self.size is not None:
            x = TF.resize(
                x,
                size=self.size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        return x

    def __len__(self):
        return len(self.dataset)


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        blueprint,
        device,
        deterministic_measurements,
        unique_seeds,
        physics,
    ):
        super().__init__()
        self.device = device
        self.deterministic_measurements = deterministic_measurements
        self.unique_seeds = unique_seeds
        self.physics_manager = getattr(physics, "__manager")

        self.ground_truth_dataset = GroundTruthDataset(
            blueprint=blueprint,
            **blueprint[GroundTruthDataset.__name__],
        )

    def __getitem__(self, index):
        x = self.ground_truth_dataset[index]
        x = x.to(self.device)

        if self.deterministic_measurements:
            if self.unique_seeds:
                seed = self.ground_truth_dataset.get_unique_id(index)
            else:
                seed = 0
        else:
            seed = None

        x = x.unsqueeze(0)
        y = self.physics_manager.randomly_degrade(x, seed=seed)
        y = y.squeeze(0)
        x = x.squeeze(0)

        return x, y

    def __len__(self):
        return len(self.ground_truth_dataset)


# NOTE: Getting small random crops should be optional and it should
# be possible to use big crops instead, e.g. to make images
# square-shaped which would enable stacking in the batch dimension.


class PrepareTrainingPairs(Module):
    def __init__(self, physics, crop_size, crop_location):
        super().__init__()
        self.physics = physics
        self.crop_size = crop_size
        self.crop_location = crop_location

    def forward(self, x, y):
        # NOTE: It'd be great if physics contained its own downsampling ratio
        # even for a blur operator.

        if self.physics.task == "sr":
            xy_size_ratio = self.physics.ratio
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
    ):
        super().__init__()
        self.synthetic_dataset = synthetic_dataset
        self.physics = physics
        self.css = css
        self.noise2inverse = noise2inverse
        self.prepare_training_pairs = prepare_training_pairs

    def __getitem__(self, index):
        x, y = self.synthetic_dataset[index]

        # NOTE: This should ideally be done in the class CSSLoss instead but
        # the border effects in the current implementation make it challenging.
        if self.css:
            physics_manager = getattr(self.physics, "__manager")
            y = y.unsqueeze(0)
            z = physics_manager.randomly_degrade(y, seed=None)
            z = z.squeeze(0)
            y = y.squeeze(0)
            x, y = y, z

        # NOTE: It's unclear why this is done only for the training set.
        # NOTE: This should ideally be done in the model.
        if self.noise2inverse:
            degradation_inverse_fn = self.physics.A_dagger
            physics_filter = getattr(self.physics, "filter", None)
            T_n2i = Noise2InverseTransform(
                task=self.physics.task,
                physics_filter=physics_filter,
                degradation_inverse_fn=degradation_inverse_fn,
            )
            x, y = T_n2i(x.unsqueeze(0), y.unsqueeze(0))
            x = x.squeeze(0)
            y = y.squeeze(0)

        # NOTE: This should ideally either be done in the model, or not at
        # all.
        x, y = self.prepare_training_pairs(x, y)

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
                f = self.physics.ratio
            else:
                f = 1
            x = TF.crop(x, top=0, left=0, height=h * f, width=w * f)

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


def get_dataset(args, purpose, physics, device):
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
        "deterministic_measurements": args.Dataset__deterministic_measure,
    }

    return Dataset(
        blueprint=blueprint,
        device=device,
        physics=physics,
        purpose=purpose,
        css=css,
        noise2inverse=noise2inverse,
    )

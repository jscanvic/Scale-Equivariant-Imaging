import torch
from torch.nn import Module
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import InterpolationMode, functional as TF
from functools import wraps

from noise2inverse import Noise2InverseTransform
from physics import Blur, Downsampling
from .crop import CropPair
from .div2k import Div2K
from .tomography import TomographyDataset
from .urban100 import Urban100
from .single_image import SingleImageDataset


class GroundTruthDataset(BaseDataset):
    def __init__(
        self, blueprint, datasets_dir, dataset, split, download, size, device, memoize_gt
    ):
        super().__init__()
        self.blueprint = blueprint
        self.datasets_dir = datasets_dir
        self.dataset = dataset
        self.split = split
        self.size = size
        self.device = device
        self.memoize_gt = memoize_gt

        if download:
            self.download(datasets_dir=self.datasets_dir, dataset=self.dataset)

    @staticmethod
    def download(datasets_dir, dataset):
        if dataset == "div2k":
            Div2K.download(datasets_dir)
        elif dataset == "urban100":
            Urban100.download(datasets_dir)
        elif dataset == "ct":
            TomographyDataset.download(datasets_dir)
        elif dataset == "single_image":
            SingleImageDataset.download(datasets_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

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
                    x = x.to("cpu")
                    cache[key] = (device, x)
                device, x = cache[key]
                x = x.to(device)
            return x

        return wrapper

    @memoize_load_image
    def __getitem__(self, index):
        if self.dataset == "div2k":
            xs = Div2K(self.split, self.datasets_dir)
        elif self.dataset == "urban100":
            xs = Urban100(self.split, self.datasets_dir)
        elif self.dataset == "ct":
            xs = TomographyDataset(self.split, self.datasets_dir)
        elif self.dataset == "single_image":
            xs = SingleImageDataset(**self.blueprint[SingleImageDataset.__name__])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        x = xs[index]
        x = x.to(self.device)
        if self.size is not None:
            x = TF.resize(
                x,
                size=self.size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        return x

    def __len__(self):
        if self.dataset == "div2k":
            xs = Div2K(split=self.split, datasets_dir=None, download=False)
            size = len(xs)
        elif self.dataset == "urban100":
            xs = Urban100(split=self.split, datasets_dir=None, download=False)
            size = len(xs)
        elif self.dataset == "ct":
            if self.split == "train":
                size = 4992
            elif self.split == "val":
                size = 100
        elif self.dataset == "single_image":
            xs = SingleImageDataset(**self.blueprint[SingleImageDataset.__name__])
            size = len(xs)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return size


# NOTE: Getting small random crops should be optional and it should
# be possible to use big crops instead, e.g. to make images
# square-shaped which would enable stacking in the batch dimension.

class PrepareTrainingPairs(Module):
    def __init__(self, physics, crop_size=48, crop_location="random"):
        super().__init__()
        self.physics = physics
        self.crop_size = crop_size
        self.crop_location = crop_location

    def forward(self, x, y):
        # NOTE: It'd be great if physics contained its own downsampling ratio
        # even for a blur operator.

        if isinstance(self.physics, Downsampling):
            xy_size_ratio = self.physics.factor
        else:
            xy_size_ratio = 1

        T_crop = CropPair(location=self.crop_location, size=self.crop_size)
        return T_crop(x, y, xy_size_ratio=xy_size_ratio)


class Dataset(BaseDataset):
    def __init__(
        self,
        blueprint,
        purpose,
        physics,
        css,
        noise2inverse,
        split,
        device,
        memoize_gt,
    ):
        super().__init__()
        self.purpose = purpose
        self.physics = physics
        self.css = css
        self.noise2inverse = noise2inverse
        # NOTE: the measurements should always be deterministic except for
        # supervised training
        self.deterministic_measurements = purpose == "test"

        self.ground_truth_dataset = GroundTruthDataset(
            blueprint=blueprint,
            device=device,
            split=split,
            memoize_gt=memoize_gt,
            **blueprint[GroundTruthDataset.__name__],
        )

        self.prepare_training_pairs = PrepareTrainingPairs(
                physics=self.physics,
                **blueprint[PrepareTrainingPairs.__name__],
            )

    def __len__(self):
        return len(self.ground_truth_dataset)

    def __getitem__(self, index):
        x = self.ground_truth_dataset[index]

        # NOTE: This should ideally be done in the class CSSLoss instead but
        # the border effects in the current implementation make it challenging.
        if self.css:
            x = self.physics(x.unsqueeze(0)).squeeze(0)

        if self.deterministic_measurements:
            # NOTE: the seed should be different for every entry
            torch.manual_seed(0)

        y = self.physics(x.unsqueeze(0)).squeeze(0)

        if self.purpose == "train":
            # NOTE: This should ideally be done in the model.
            if self.noise2inverse:
                T_n2i = Noise2InverseTransform(self.physics)
                x, y = T_n2i(x.unsqueeze(0), y.unsqueeze(0))
                x = x.squeeze(0)
                y = y.squeeze(0)


            # NOTE: This should ideally either be done in the model, or not at
            # all.
            x, y = self.prepare_training_pairs(x, y)
        elif self.purpose == "test":
            # NOTE: This should ideally be removed.
            if self.noise2inverse:
                # bug fix: make y have even height and width
                if isinstance(self.physics, Blur):
                    w = 2 * (y.shape[1] // 2)
                    h = 2 * (y.shape[2] // 2)
                    y = y[:, :w, :h]

            # NOTE: This should ideally be removed.
            # crop x to make its dimensions be a multiple of u's dimensions
            if x.shape != y.shape:
                h, w = y.shape[1], y.shape[2]
                if isinstance(self.physics, Downsampling):
                    f = self.physics.factor
                else:
                    f = 1
                x = TF.crop(x, top=0, left=0, height=h * f, width=w * f)
        else:
            raise ValueError(f"Unknown purpose: {self.purpose}")

        return x, y


def get_dataset(args, purpose, physics, device):
    if purpose == "test":
        noise2inverse = args.noise2inverse
        css = False
        split = args.split
        memoize_gt = False
    elif purpose == "train":
        noise2inverse = args.method == "noise2inverse"
        css = args.method == "css"
        split = "train"
        memoize_gt = args.memoize_gt
    else:
        raise ValueError(f"Unknown purpose: {purpose}")

    blueprint = {}

    blueprint[GroundTruthDataset.__name__] = {
            # NOTE: This argument should be named according to the class
            # GroundTruthDataset but happens to be used (wrongly) elsewhere and
            # this must be dealt with first
            "dataset": args.dataset,
            "datasets_dir": args.GroundTruthDataset__datasets_dir,
            "download": args.GroundTruthDataset__download,
            "size": args.GroundTruthDataset__size,
        }

    blueprint[PrepareTrainingPairs.__name__] = {
            "crop_size": args.PrepareTrainingPairs__crop_size,
            "crop_location": args.PrepareTrainingPairs__crop_location,
        }


    blueprint[SingleImageDataset.__name__] = {
            "image_path": args.SingleImageDataset__image_path,
            "duplicates_count": args.SingleImageDataset__duplicates_count,
        }

    return Dataset(
            blueprint=blueprint,
            device=device,
            physics=physics,
            purpose=purpose,
            css=css,
            noise2inverse=noise2inverse,
            split=split,
            memoize_gt=memoize_gt,
    )

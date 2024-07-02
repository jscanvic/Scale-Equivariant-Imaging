import torch
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import InterpolationMode, functional as TF
from functools import wraps

from noise2inverse import Noise2InverseTransform
from physics import Blur, Downsampling
from .crop import CropPair
from .div2k import Div2K
from .tomography import TomographyDataset
from .urban100 import Urban100


class GroundTruthDataset(BaseDataset):
    def __init__(
        self, datasets_dir, dataset, split, download, resize, device, memoize_gt
    ):
        super().__init__()
        self.datasets_dir = datasets_dir
        self.dataset = dataset
        self.split = split
        self.resize = resize
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
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        x = xs[index]
        x = x.to(self.device)
        if self.resize is not None:
            x = TF.resize(
                x,
                size=self.resize,
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
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        return size


class Dataset(BaseDataset):
    def __init__(
        self,
        purpose,
        physics,
        css,
        noise2inverse,
        fixed_seed,
        datasets_dir,
        dataset,
        split,
        download,
        resize,
        device,
        memoize_gt,
    ):
        super().__init__()
        self.purpose = purpose
        self.physics = physics
        self.css = css
        self.noise2inverse = noise2inverse
        self.fixed_seed = fixed_seed

        self.ground_truth_dataset = GroundTruthDataset(
            datasets_dir=datasets_dir,
            dataset=dataset,
            split=split,
            download=download,
            resize=resize,
            device=device,
            memoize_gt=memoize_gt,
        )

    def __len__(self):
        return len(self.ground_truth_dataset)

    def __getitem__(self, index):
        x = self.ground_truth_dataset[index]

        if self.css:
            x = self.physics(x.unsqueeze(0)).squeeze(0)

        if self.fixed_seed:
            torch.manual_seed(0)

        y = self.physics(x.unsqueeze(0)).squeeze(0)

        if self.purpose == "train":
            if self.noise2inverse:
                T_n2i = Noise2InverseTransform(self.physics)
                x, y = T_n2i(x.unsqueeze(0), y.unsqueeze(0))
                x = x.squeeze(0)
                y = y.squeeze(0)

            if isinstance(self.physics, Downsampling):
                xy_size_ratio = self.physics.factor
            else:
                xy_size_ratio = 1

            T_crop = CropPair("random", 48)
            x, y = T_crop(x, y, xy_size_ratio=xy_size_ratio)
        elif self.purpose == "test":
            if self.noise2inverse:
                # bug fix: make y have even height and width
                if isinstance(self.physics, Blur):
                    w = 2 * (y.shape[1] // 2)
                    h = 2 * (y.shape[2] // 2)
                    y = y[:, :w, :h]

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
    resize = args.gt_size if args.resize_gt else None
    if purpose == "test":
        noise2inverse = args.noise2inverse
        css = False
        fixed_seed = True
        split = args.split
        memoize_gt = False
    elif purpose == "train":
        noise2inverse = args.method == "noise2inverse"
        css = args.method == "css"
        fixed_seed = False
        split = "train"
        memoize_gt = args.memoize_gt

    blueprint = {}
    blueprint[Dataset.__name__] = {
            "datasets_dir": args.datasets_dir,
            "dataset": args.dataset,
            "download": args.download,
        }

    return Dataset(
            device=device,
            physics=physics,
            purpose=purpose,
            css=css,
            noise2inverse=noise2inverse,
            fixed_seed=fixed_seed,
            resize=resize,
            split=split,
            memoize_gt=memoize_gt,
            **blueprint[Dataset.__name__],
    )

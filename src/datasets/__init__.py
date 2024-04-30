import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF
from functools import wraps

from noise2inverse import Noise2InverseTransform
from physics import Blur, Downsampling
from .crop import CropPair
from .div2k import Div2K
from .tomography import TomographyDataset
from .urban100 import Urban100


def download_dataset(datasets_dir, dataset="div2k"):
    """Download DIV2K dataset"""
    assert dataset in ["div2k", "urban100", "ct"]
    if dataset == "div2k":
        Div2K.download(datasets_dir)
    elif dataset == "urban100":
        Urban100.download(datasets_dir)
    elif dataset == "ct":
        TomographyDataset.download(datasets_dir)

@staticmethod
def _memoize_load_image(f):
    cache = {}

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not kwargs["memoize"]:
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

@_memoize_load_image
def load_image(index, split, datasets_dir, resize, device="cpu", dataset="div2k", force_rgb=False, memoize=False):
    assert dataset in ["div2k", "urban100", "ct"]
    if dataset == "div2k":
        xs = Div2K(split, datasets_dir)
    elif dataset == "urban100":
        xs = Urban100(split, datasets_dir)
    elif dataset == "ct":
        channels = 3 if force_rgb else 1
        xs = TomographyDataset(split, datasets_dir, channels=channels)

    x = xs[index]
    x = x.to(device)
    if resize is not None:
        x = TF.resize(
            x,
            size=resize,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    return x


class TrainingDataset(Dataset):
    """
    Training dataset used in the paper

    :param str root: root directory of the dataset
    :param deepinv.physics.Physics physics: forward model
    :param int resize: resize the ground truth images to this size
    :param bool css: to be enabled for CSS training
    :param bool download: download the dataset
    :param str device: device to use
    """

    def __init__(self, root, physics, resize, css=False, download=False, device="cpu", dataset="div2k", force_rgb=False,
                 method=None, memoize_gt=False):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.css = css
        self.device = device
        self.force_rgb = force_rgb
        self.method = method
        self.memoize_gt = memoize_gt

        assert dataset in ["div2k", "urban100", "ct"]
        self.dataset = dataset

        if download:
            download_dataset(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        x = load_image(index, "train", self.root, self.resize, self.device, dataset=self.dataset,
                       force_rgb=self.force_rgb, memoize=self.memoize_gt)

        if self.css:
            x = self.physics(x.unsqueeze(0)).squeeze(0)

        y = self.physics(x.unsqueeze(0)).squeeze(0)

        if self.method == "noise2inverse":
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
        return x, y

    def __len__(self):
        if self.dataset == "div2k":
            size = 800
        elif self.dataset == "urban100":
            size = 90
        elif self.dataset == "ct":
            size = 4992
        return size


class EvalDataset(Dataset):
    """
    Evaluation dataset used in the paper

    :param str root: root directory of the dataset
    :param deepinv.physics.Physics physics: forward model
    :param int resize: resize the ground truth images to this size
    :param bool download: download the dataset
    :param str device: device to use
    """

    def __init__(self,
                 root,
                 physics,
                 resize,
                 download=False,
                 device="cpu",
                 dataset="div2k",
                 force_rgb=False,
                 method=None,
                 memoize_gt=False,):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.device = device
        self.dataset = dataset
        self.force_rgb = force_rgb
        self.method = method
        self.memoize_gt = memoize_gt

        if download:
            download_dataset(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        x = load_image(index, "val", self.root, self.resize, self.device, dataset=self.dataset,
                       force_rgb=self.force_rgb, memoize=self.memoize_gt)
        y = self.physics(x.unsqueeze(0)).squeeze(0)

        if isinstance(self.physics, Downsampling):
            xy_size_ratio = self.physics.factor
        else:
            xy_size_ratio = 1

        T_crop = CropPair("center", 48)
        x, y = T_crop(x, y, xy_size_ratio=xy_size_ratio)
        return x, y

    def __len__(self):
        return 10


class TestDataset(Dataset):
    """
    Test dataset used in the paper

    :param str root: root directory of the dataset
    :param str split: split to use (i.e. train or val)
    :param deepinv.physics.Physics physics: forward model
    :param int resize: resize the ground truth images to this size
    :param str device: device to use
    :param bool download: download the dataset
    """

    def __init__(
            self,
            root,
            split,
            physics,
            resize=None,
            device="cpu",
            download=False,
            dataset="div2k",
            max_size=None,
            force_rgb=False,
            offset=None,
            method=None,
            memoize_gt=False,
    ):
        self.resize = resize
        self.split = split
        self.root = root
        self.physics = physics
        self.device = device
        self.dataset = dataset
        self.max_size = max_size
        self.force_rgb = force_rgb
        self.offset = offset
        self.method = method
        self.memoize_gt = memoize_gt

        if download:
            download_dataset(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        if self.offset is not None:
            index += self.offset
        x = load_image(index, self.split, self.root, self.resize, self.device, dataset=self.dataset,
                       force_rgb=self.force_rgb, memoize=self.memoize_gt)

        torch.manual_seed(0)
        y = self.physics(x.unsqueeze(0)).squeeze(0)

        # bug fix: make y have even height and width
        if isinstance(self.physics, Blur) and self.method == "noise2inverse":
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
        return x, y

    def __len__(self):
        if self.dataset == "div2k":
            max_size = 100 if self.max_size is None else self.max_size
            if self.split == "train":
                size = min(800, max_size)
            elif self.split == "val":
                size = min(100, max_size)
        elif self.dataset == "urban100":
            if self.split == "train":
                size = 90
            elif self.split == "val":
                size = 10
        elif self.dataset == "ct":
            max_size = 100 if self.max_size is None else self.max_size
            if self.split == "train":
                size = min(4992, max_size)
            elif self.split == "val":
                size = min(100, max_size)
        return size

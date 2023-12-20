from math import ceil

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.transforms.functional import to_tensor


def minsize_pad(x, size, padding_mode="constant", fill=0):
    """Pad an image to a minimum size"""
    h_padding = max(0, size - x.shape[1])
    w_padding = max(0, size - x.shape[2])
    return TF.pad(x, [0, 0, w_padding, h_padding], padding_mode=padding_mode, fill=fill)


def get_random_patch_pair(x, y, size=48):
    """Get a random patch pair from a pair of images"""
    f = int(ceil(x.shape[1] / y.shape[1]))

    x = minsize_pad(x, size * f)
    y = minsize_pad(y, size)

    h, w = y.shape[-2:]

    i = torch.randint(0, h - size + 1, size=(1,)).item()
    j = torch.randint(0, w - size + 1, size=(1,)).item()

    x_crop = TF.crop(x, top=i * f, left=j * f, height=size * f, width=size * f)
    y_crop = TF.crop(y, top=i, left=j, height=size, width=size)

    return x_crop, y_crop


def get_center_patch_pair(x, y, size=48):
    """Get the center patch pair from a pair of images"""
    f = int(ceil(x.shape[1] / y.shape[1]))

    x = minsize_pad(x, size * f)
    y = minsize_pad(y, size)

    h, w = y.shape[-2:]

    i = (h - size) // 2
    j = (w - size) // 2

    x_patch = TF.crop(x, top=i * f, left=j * f, height=size * f, width=size * f)
    y_patch = TF.crop(y, top=i, left=j, height=size, width=size)

    return x_patch, y_patch


def download_div2k(root):
    """Download DIV2K dataset"""
    archives = [
        (
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "bdc2d9338d4e574fe81bf7d158758658",
        ),
        (
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "9fcdda83005c5e5997799b69f955ff88",
        ),
    ]

    for url, md5 in archives:
        download_and_extract_archive(url, f"{root}/DIV2K", md5=md5)


def load_image(index, split, root, resize, device="cpu"):
    """Load an image from the DIV2K dataset"""
    index = index + 1 if split == "train" else index + 801

    if split == "train":
        split_root = f"{root}/DIV2K/DIV2K_train_HR"
    else:
        split_root = f"{root}/DIV2K/DIV2K_valid_HR"

    file_path = f"{split_root}/{index:04d}.png"
    x = Image.open(file_path)
    x = to_tensor(x)
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

    def __init__(self, root, physics, resize, css=False, download=False, device="cpu"):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.css = css
        self.device = device

        if download:
            download_div2k(self.root)

    def __getitem__(self, index):
        x = load_image(index, "train", self.root, self.resize, self.device)

        if self.css:
            x = self.physics(x.unsqueeze(0)).squeeze(0)

        y = self.physics(x.unsqueeze(0)).squeeze(0)
        x_patch, y_patch = get_random_patch_pair(x, y)
        return x_patch, y_patch

    def __len__(self):
        return 800


class EvalDataset(Dataset):
    """
    Evaluation dataset used in the paper

    :param str root: root directory of the dataset
    :param deepinv.physics.Physics physics: forward model
    :param int resize: resize the ground truth images to this size
    :param bool download: download the dataset
    :param str device: device to use
    """

    def __init__(self, root, physics, resize, download=False, device="cpu"):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.device = device

        if download:
            download_div2k(self.root)

    def __getitem__(self, index):
        x = load_image(index, "val", self.root, self.resize, self.device)
        y = self.physics(x.unsqueeze(0)).squeeze(0)
        x_patch, y_patch = get_center_patch_pair(x, y)
        return x_patch, y_patch

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
    ):
        self.resize = resize
        self.split = split
        self.root = root
        self.physics = physics
        self.device = device

        if download:
            download_div2k(self.root)

    def __getitem__(self, index):
        x = load_image(index, self.split, self.root, self.resize, self.device)

        torch.manual_seed(0)
        y = self.physics(x.unsqueeze(0)).squeeze(0)

        # crop x to make its dimensions be a multiple of u's dimensions
        if x.shape != y.shape:
            h, w = y.shape[2], y.shape[3]
            f = int(ceil(x.shape[2] / y.shape[2]))
            x = TF.crop(x, top=0, left=0, height=h * f, width=w * f)

        return x, y

    def __len__(self):
        return 10

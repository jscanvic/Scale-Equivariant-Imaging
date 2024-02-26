from math import ceil

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import InterpolationMode, functional as TF
from torchvision.transforms.functional import to_tensor
from deepinv.datasets import HDF5Dataset
from noise2inverse import n2i_pair


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


def download_div2k(root, dataset="div2k"):
    """Download DIV2K dataset"""
    assert dataset in ["div2k", "urban100", "ct"]
    if dataset == "div2k":
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
    elif dataset == "urban100":
        download_and_extract_archive(
                "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true",
                f"{root}/Urban100",
                filename="Urban100_HR.tar.gz",
                md5="65d9d84a34b72c6f7ca1e26a12df1e4c"
        )
    elif dataset == "ct":
        download_and_extract_archive(
                "https://huggingface.co/jtachella/equivariant_bootstrap/resolve/main/Tomography/dinv_dataset0.h5?download=true",
                f"{root}/CT",
                filename="dinv_dataset0.h5",
                md5="6f1e1a4f5c0a1f0e0a4b6e6b9a8b4b4b"
        )


def load_image(index, split, root, resize, device="cpu", dataset="div2k", force_rgb=False):
    """Load an image from the DIV2K dataset"""
    assert dataset in ["div2k", "urban100", "ct"]
    if dataset == "div2k":
        index = index + 1 if split == "train" else index + 801

        if split == "train":
            split_root = f"{root}/DIV2K/DIV2K_train_HR"
        else:
            split_root = f"{root}/DIV2K/DIV2K_valid_HR"

        file_path = f"{split_root}/{index:04d}.png"
    elif dataset == "urban100":
        index = index + 1 if split == "train" else index + 91
        file_path = f"{root}/Urban100/Urban100_HR/img_{index:03d}.png"

    if dataset in ["div2k", "urban100"]:
        x = Image.open(file_path)
        x = to_tensor(x)
    elif dataset == "ct":
        is_train_split = split == "train"
        ims = HDF5Dataset(f"{root}/CT/dinv_dataset0.h5", train=is_train_split)
        x, _ = ims[index]
        if force_rgb:
            x = x.repeat(3, 1, 1)


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

    def __init__(self, root, physics, resize, css=False, download=False, device="cpu", dataset="div2k", force_rgb=False, method=None, patch_size=48):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.css = css or method == "css"
        self.method = method
        self.device = device
        self.force_rgb = force_rgb
        self.patch_size = patch_size

        assert dataset in ["div2k", "urban100", "ct"]
        self.dataset = dataset

        if download:
            download_div2k(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        x = load_image(index, "train", self.root, self.resize, self.device, dataset=self.dataset, force_rgb=self.force_rgb)

        if self.css:
            x = self.physics(x.unsqueeze(0)).squeeze(0)

        y = self.physics(x.unsqueeze(0)).squeeze(0)

        from torchvision.utils import save_image

        for j in range(x.shape[0]):
            save_image(x[j], f"__debug/x_{j}.png")

        for j in range(y.shape[0]):
            save_image(y[j], f"__debug/y_{j}.png")

        if self.method == "noise2inverse":
            x, y = n2i_pair(y.unsqueeze(0)).squeeze(0)

        for j in range(y.shape[0]):
            save_image(y[j], f"__debug/inp_{j}.png")

        for j in range(x.shape[0]):
            save_image(x[j], f"__debug/tgt_{j}.png")

        return get_random_patch_pair(x, y, size=48)

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

    def __init__(self, root, physics, resize, download=False, device="cpu", dataset="div2k", force_rgb=False):
        super().__init__()
        self.root = root
        self.physics = physics
        self.resize = resize
        self.device = device
        self.dataset = dataset
        self.force_rgb = force_rgb

        if download:
            download_div2k(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        x = load_image(index, "val", self.root, self.resize, self.device, dataset=self.dataset, force_rgb=self.force_rgb)
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
        dataset="div2k",
        max_size=None,
        force_rgb=False,
        offset=None,
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

        if download:
            download_div2k(self.root, dataset=self.dataset)

    def __getitem__(self, index):
        if self.offset is not None:
            index += self.offset
        x = load_image(index, self.split, self.root, self.resize, self.device, dataset=self.dataset, force_rgb=self.force_rgb)

        torch.manual_seed(0)
        y = self.physics(x.unsqueeze(0)).squeeze(0)

        # crop x to make its dimensions be a multiple of u's dimensions
        if x.shape != y.shape:
            h, w = y.shape[1], y.shape[2]
            f = int(ceil(x.shape[1] / y.shape[1]))
            x = TF.crop(x, top=0, left=0, height=h * f, width=w * f)

        return x, y

    def __len__(self):
        if self.dataset == "div2k":
            max_size = 10 if self.max_size is None else self.max_size
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

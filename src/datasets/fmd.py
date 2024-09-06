import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import read_image as base_read_image

from glob import glob
from os.path import basename


def read_image(file_path):
    x = base_read_image(file_path)
    x = x.to(torch.float) / 255.0
    if x.size(0) == 4:
        x = x[:3]
    elif x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    elif x.size(0) != 3:
        raise ValueError(f"Unexpected number of channels: {x.size(0)}")
    return x


class FMD(Dataset):
    def __init__(self, split, datasets_dir, download=False):
        super().__init__()

        train_gt_paths = glob(f"{datasets_dir}/Split_FMD/train/*.png")
        val_gt_paths = glob(f"{datasets_dir}/Split_FMD/test/*.png")
        train_split_size = len(train_gt_paths)
        val_split_size = len(val_gt_paths)

        # Integrity check
        assert val_split_size == 24
        assert train_split_size == 216

        if split == "train":
            self.split_offset = 0
            self.split_size = train_split_size
            self.gt_paths = train_gt_paths
        elif split == "val":
            self.split_offset = train_split_size
            self.split_size = val_split_size
            self.gt_paths = val_gt_paths
        else:
            raise ValueError(f"Unknown split: {split}")

        if download:
            self.download(datasets_dir)

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        x = read_image(gt_path)
        return x

    def __len__(self):
        return self.split_size

    def get_unique_id(self, index):
        return self.split_offset + index

    @staticmethod
    def download(datasets_dir):
        raise NotImplementedError()

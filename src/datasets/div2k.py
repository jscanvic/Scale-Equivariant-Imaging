import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import read_image


class Div2K(Dataset):
    def __init__(self, split, datasets_dir, download=False):
        super().__init__()
        self.datasets_dir = datasets_dir

        assert split in ["train", "val"]
        self.split = split
        if self.split == "train":
            self.split_root = f"{self.datasets_dir}/DIV2K/DIV2K_train_HR"
            self.split_offset = 1
            self.split_size = 800
        elif self.split == "val":
            self.split_root = f"{self.datasets_dir}/DIV2K/DIV2K_valid_HR"
            self.split_offset = 801
            self.split_size = 100

        if download:
            self.download(datasets_dir)

    def __getitem__(self, index):
        index = self.split_offset + index
        file_path = f"{self.split_root}/{index:04d}.png"
        x = read_image(file_path)
        x = x.to(torch.float) / 255.0
        return x

    def __len__(self):
        return self.split_size

    @staticmethod
    def download(datasets_dir):
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
            download_and_extract_archive(url, f"{datasets_dir}/DIV2K", md5=md5)

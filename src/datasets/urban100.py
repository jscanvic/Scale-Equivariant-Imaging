import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import read_image


class Urban100(Dataset):
    def __init__(self, split, datasets_dir, download=False):
        super().__init__()
        self.datasets_dir = datasets_dir

        assert split in ["train", "val"]
        self.split = split
        if self.split == "train":
            self.split_offset = 1
            self.split_size = 90
        elif self.split == "val":
            self.split_offset = 91
            self.split_size = 10

        if download:
            self.download(datasets_dir)

    def __getitem__(self, index):
        index = self.split_offset + index
        file_path = f"{self.datasets_dir}/Urban100/Urban100_HR/img_{index:03d}.png"
        x = read_image(file_path)
        x = x.to(torch.float) / 255.
        return x

    def __len__(self):
        return self.split_size

    @staticmethod
    def download(datasets_dir):
        download_and_extract_archive(
            "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true",
            f"{datasets_dir}/Urban100",
            filename="Urban100_HR.tar.gz",
            md5="65d9d84a34b72c6f7ca1e26a12df1e4c"
        )

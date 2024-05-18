from deepinv.datasets import HDF5Dataset
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class TomographyDataset(Dataset):
    def __init__(self, split, datasets_dir, channels=3, download=False):
        super().__init__()
        assert split in ["train", "val"]
        self.split = split
        self.datasets_dir = datasets_dir
        assert channels in [1, 3], "Channels must be 1 or 3"
        self.channels = channels

        self._dataset = HDF5Dataset(
            f"{self.datasets_dir}/CT/dinv_dataset0.h5", train=(self.split == "train")
        )

        if download:
            self.download(datasets_dir)

    def __getitem__(self, index):
        x, _ = self._dataset[index]
        if self.channels == 3:
            x = x.repeat(3, 1, 1)
        return x

    def __len__(self):
        return len(self._dataset)

    @staticmethod
    def download(datasets_dir):
        download_and_extract_archive(
            "https://huggingface.co/jtachella/equivariant_bootstrap/resolve/main/Tomography/dinv_dataset0.h5?download=true",
            f"{datasets_dir}/CT",
            filename="dinv_dataset0.h5",
            md5="6f1e1a4f5c0a1f0e0a4b6e6b9a8b4b4b",
        )

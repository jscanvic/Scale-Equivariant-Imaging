import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class SingleImageDataset(Dataset):
    # NOTE: For the sake of simplicity, the dataset is made to contain many
    # duplicates of the same image, enabling to train for the same amount of
    # epochs and to save checkpoints at the same frequency as for other
    # datasets.
    def __init__(self, image_path, duplicates_count, download=False):
        self.duplicates_count = duplicates_count
        self.image_path = image_path
        self.im = None

        if download:
            self.download(datasets_dir)


    def __len__(self):
        return self.duplicates_count

    def __getitem__(self, idx):
        if self.im is None:
            im = read_image(self.image_path)
            im = im.to(torch.float) / 255.0
            self.im = im
        return self.im

    @staticmethod
    def download(datasets_dir):
        pass

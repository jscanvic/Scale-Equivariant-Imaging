from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, functional as TF
from functools import wraps

from .div2k import Div2K
from .tomography import TomographyDataset
from .urban100 import Urban100
from .fmd import FMD
from .single_image import SingleImageDataset


class GroundTruthDataset(Dataset):
    def __init__(
        self,
        blueprint,
        datasets_dir,
        dataset_name,
        split,
        download,
        size,
        memoize_gt,
    ):
        super().__init__()
        self.size = size
        self.memoize_gt = memoize_gt

        dataset_name = dataset_name.lower()

        if dataset_name == "div2k":
            self.dataset = Div2K(split, datasets_dir, download=download)
        elif dataset_name == "urban100":
            self.dataset = Urban100(split, datasets_dir, download=download)
        elif dataset_name == "ct":
            self.dataset = TomographyDataset(split, datasets_dir, download=download)
        elif dataset_name == "fmd":
            self.dataset = FMD(split, datasets_dir, download=download)
        elif dataset_name == "single_image":
            self.dataset = SingleImageDataset(**blueprint[SingleImageDataset.__name__])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_unique_id(self, index):
        if hasattr(self.dataset, "get_unique_id"):
            id = self.dataset.get_unique_id(index)
        else:
            id = index
        return id

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
                    # NOTE: This should be unnecessary.
                    x = x.to("cpu")
                    cache[key] = (device, x)
                device, x = cache[key]
                x = x.to(device)
            return x

        return wrapper

    @memoize_load_image
    def __getitem__(self, index):
        x = self.dataset[index]
        if self.size is not None:
            x = TF.resize(
                x,
                size=self.size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        return x

    def __len__(self):
        return len(self.dataset)

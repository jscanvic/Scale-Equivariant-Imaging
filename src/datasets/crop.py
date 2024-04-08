import torch
from math import ceil

from torch.nn import Module
from torchvision.transforms import functional as TF


class CropPair(Module):
    def __init__(self, location, size=48):
        super().__init__()
        assert location in ["random", "center"]
        self.location = location
        self.size = size

    def forward(self, x, y, xy_size_ratio=None):
        if xy_size_ratio is None:
            xy_size_ratio = int(ceil(x.shape[1] / y.shape[1]))
        T_pad_x = MinSizePadding(self.size * xy_size_ratio, padding_mode="constant", fill=0)
        x = T_pad_x(x)
        T_pad_y = MinSizePadding(self.size, padding_mode="constant", fill=0)
        y = T_pad_y(y)
        h, w = y.shape[-2:]
        if self.location == "random":
            i = torch.randint(0, h - self.size + 1, size=(1,)).item()
            j = torch.randint(0, w - self.size + 1, size=(1,)).item()
        elif self.location == "center":
            i = (h - self.size) // 2
            j = (w - self.size) // 2
        x_crop = TF.crop(x, top=i * xy_size_ratio, left=j * xy_size_ratio, height=self.size * xy_size_ratio,
                         width=self.size * xy_size_ratio)
        y_crop = TF.crop(y, top=i, left=j, height=self.size, width=self.size)
        return x_crop, y_crop


class MinSizePadding(Module):
    def __init__(self, size, padding_mode="constant", fill=0):
        super().__init__()
        self.size = size
        self.padding_mode = padding_mode
        self.fill = fill

    def forward(self, x):
        h_padding = max(0, self.size - x.shape[1])
        w_padding = max(0, self.size - x.shape[2])
        return TF.pad(x,
                      [0, 0, w_padding, h_padding],
                      padding_mode=self.padding_mode,
                      fill=self.fill)

from torch.nn import Module
from torch.nn.functional import interpolate


class Upsample(Module):
    def __init__(self, factor=2, interpolation_mode="bicubic"):
        super().__init__()
        self.factor = factor
        assert interpolation_mode == "bicubic", "Only bicubic upsampling is supported"
        self.mode = interpolation_mode


    def forward(self, y):
        return interpolate(y, scale_factor=self.factor, mode=self.mode)

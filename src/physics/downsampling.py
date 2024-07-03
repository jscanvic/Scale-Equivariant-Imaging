import torch
from deepinv.physics import LinearPhysics
from torch.nn import Module
from torch.nn.functional import interpolate


class Downsampling(LinearPhysics):
    def __init__(self, ratio, antialias):
        super().__init__()
        self.ratio = ratio
        self.antialias = antialias

    def A(self, x):
        return interpolate(x,
                           scale_factor=1 / self.ratio,
                           mode="bicubic",
                           antialias=self.antialias)

    # NOTE: This should be the true adjoint.
    def A_adjoint(self, y):
        return imresize(y, scale=self.ratio)

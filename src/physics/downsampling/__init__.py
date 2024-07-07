import torch
from deepinv.physics import LinearPhysics
from torch.nn import Module
from torch.nn.functional import interpolate

from deepinv.physics import adjoint_function


class Downsampling(LinearPhysics):
    def __init__(self, ratio, antialias, true_adjoint=False):
        super().__init__()
        self.ratio = ratio
        self.antialias = antialias
        self.true_adjoint = true_adjoint

    def A(self, x):
        return interpolate(
            x, scale_factor=1 / self.ratio, mode="bicubic", antialias=self.antialias
        )

    def A_adjoint(self, y):
        if self.true_adjoint:
            # NOTE: It'd be better to avoid the intermediate callable variable.
            input_size = (
                y.shape[0],
                y.shape[1],
                y.shape[2] * self.ratio,
                y.shape[3] * self.ratio,
            )
            A_adjoint = adjoint_function(self.A, input_size=input_size, device=y.device)
            x = A_adjoint(y)
        else:
            # NOTE: This is deprecated.
            x = interpolate(y, scale_factor=self.ratio, mode="bicubic")
        return x

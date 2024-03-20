# code adapted from
# https://raw.githubusercontent.com/ahendriksen/noise2inverse/master/noise2inverse/datasets.py

import numpy as np
import torch
from deepinv.physics import Blur
from itertools import combinations

from torch.nn import Module


class ImageSlices(Module):
    def __init__(self, physics, num_splits=4):
        super().__init__()
        self.num_splits = num_splits
        if isinstance(physics, Blur):
            # we use a faster implementation than the default one
            self.backproject = InverseFilter(physics)
        else:
            self.backproject = physics.A_dagger

    def forward(self, y):
        measurement_slices = self.measurement_slices(y)
        image_slices = [
            self.backproject(measurement_slice)
            for measurement_slice in measurement_slices
        ]
        return image_slices

    def measurement_slices(self, y):

        measurement_slices = []

        for j in range(self.num_splits):
            measurement_slice = torch.zeros_like(y)
            measurement_slice[:, :, j::self.num_splits, :] = y[:, :, j::self.num_splits, :]
            measurement_slices.append(measurement_slice)

        return measurement_slices


class InverseFilter(Module):
    def __init__(self, physics):
        super().__init__()
        assert isinstance(physics, Blur)
        assert physics.filter.dim() == 4
        self.kernel = physics.filter.squeeze(0).squeeze(0)

    def forward(self, y):
        assert y.dim() == 4
        psf = torch.zeros(
            (y.shape[-2], y.shape[-1]),
            device=y.device,
            dtype=y.dtype
        )
        psf[: self.kernel.shape[-2], : self.kernel.shape[-1]] = self.kernel
        psf = torch.roll(psf,
                         (-(self.kernel.shape[-2] // 2), -(self.kernel.shape[-1] // 2)),
                         dims=(-2, -1))

        s = (y.shape[-2], y.shape[-1])
        x_hat = torch.fft.rfft2(y, dim=(-2, -1))

        otf = torch.fft.rfft2(psf, dim=(-2, -1))
        otf = otf.expand(x_hat.shape)
        x_hat = x_hat / otf

        return torch.fft.irfft2(x_hat, dim=(-2, -1), s=s)


class Noise2InverseModel(Module):
    def __init__(self, backbone, physics, num_splits=4, strategy="X:1"):
        super().__init__()
        self.physics = physics
        self.backbone = backbone
        self.num_splits = num_splits
        self.stragegy = strategy

    def forward(self, y):
        inputs = self.compute_inputs(y)
        reconstructions = [self.backbone(input) for input in inputs]
        x_hat = torch.stack(reconstructions).mean(dim=0)
        return x_hat

    def compute_inputs(self, y):
        T = ImageSlices(self.physics, num_splits=self.num_splits)
        image_slices = T(y)
        if self.stragegy == "X:1":
            num_input = self.num_splits - 1
        else:
            num_input = 1
        split_idxs = set(range(self.num_splits))
        input_idxs = list(combinations(split_idxs, num_input))
        inps = []
        for idx in range(len(input_idxs)):
            inputs = [image_slices[j] for j in input_idxs[idx]]
            inp = torch.sum(torch.stack(inputs), dim=0)
            inps.append(inp)
        return inps


def n2i_pair(y, physics, strategy="X:1", num_splits=4):
    T = ImageSlices(physics, num_splits=num_splits)
    image_slices = T(y)

    if strategy == "X:1":
        num_input = num_splits - 1
    else:
        num_input = 1

    split_idxs = set(range(num_splits))
    input_idxs = list(combinations(split_idxs, num_input))
    target_idxs = [split_idxs - set(idxs) for idxs in input_idxs]

    idx = np.random.randint(0, len(input_idxs))

    inputs = [image_slices[j] for j in input_idxs[idx]]
    targets = [image_slices[j] for j in target_idxs[idx]]

    inp = torch.sum(torch.stack(inputs), dim=0)
    tgt = torch.sum(torch.stack(targets), dim=0)

    return tgt, inp


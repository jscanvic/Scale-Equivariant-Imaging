# code adapted from
# https://raw.githubusercontent.com/ahendriksen/noise2inverse/master/noise2inverse/datasets.py

import numpy as np
import torch
from itertools import combinations

from torch.nn import Module


class ImageSlices(Module):
    def __init__(self, num_splits, task, physics_filter, degradation_inverse_fn):
        super().__init__()
        self.num_splits = num_splits
        if task == "deblurring":
            # we use a faster implementation than the default one
            kernel = physics_filter
            assert kernel is not None
            assert kernel.dim() == 4
            kernel = kernel.squeeze(0).squeeze(0)
            self.backproject = InverseFilter(kernel=kernel)
        else:
            self.backproject = degradation_inverse_fn

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
            measurement_slice[:, :, j :: self.num_splits, :] = y[
                :, :, j :: self.num_splits, :
            ]
            measurement_slices.append(measurement_slice)

        return measurement_slices


class InverseFilter(Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, y):
        assert y.dim() == 4
        psf = torch.zeros((y.shape[-2], y.shape[-1]), device=y.device, dtype=y.dtype)
        psf[: self.kernel.shape[-2], : self.kernel.shape[-1]] = self.kernel
        psf = torch.roll(
            psf,
            (-(self.kernel.shape[-2] // 2), -(self.kernel.shape[-1] // 2)),
            dims=(-2, -1),
        )

        s = (y.shape[-2], y.shape[-1])
        x_hat = torch.fft.rfft2(y, dim=(-2, -1))

        otf = torch.fft.rfft2(psf, dim=(-2, -1))
        otf = otf.expand(x_hat.shape)
        x_hat = x_hat / otf

        return torch.fft.irfft2(x_hat, dim=(-2, -1), s=s)


class Noise2InverseModel(Module):
    def __init__(
        self,
        backbone,
        task,
        physics_filter,
        degradation_inverse_fn,
        num_splits=4,
        strategy="X:1",
    ):
        super().__init__()
        self.backbone = backbone
        self.num_splits = num_splits
        self.stragegy = strategy
        self.transform = ImageSlices(
            num_splits=self.num_splits,
            task=task,
            physics_filter=physics_filter,
            degradation_inverse_fn=degradation_inverse_fn,
        )

    def forward(self, y):
        inputs = self.compute_inputs(y)
        reconstructions = [self.backbone(input) for input in inputs]
        x_hat = torch.stack(reconstructions).sum(dim=0)
        return x_hat

    def compute_inputs(self, y):
        image_slices = self.transform(y)
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


class Noise2InverseTransform(Module):
    def __init__(
        self,
        task,
        physics_filter,
        degradation_inverse_fn,
        strategy="X:1",
        num_splits=4,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_splits = num_splits
        self.transform = ImageSlices(
            num_splits=self.num_splits,
            task=task,
            physics_filter=physics_filter,
            degradation_inverse_fn=degradation_inverse_fn,
        )

    def forward(self, x, y):
        image_slices = self.transform(y)
        if self.strategy == "X:1":
            num_input = self.num_splits - 1
        else:
            num_input = 1
        split_idxs = set(range(self.num_splits))
        input_idxs = list(combinations(split_idxs, num_input))
        target_idxs = [split_idxs - set(idxs) for idxs in input_idxs]
        idx = np.random.randint(0, len(input_idxs))
        inputs = [image_slices[j] for j in input_idxs[idx]]
        targets = [image_slices[j] for j in target_idxs[idx]]
        inp = torch.sum(torch.stack(inputs), dim=0)
        tgt = torch.sum(torch.stack(targets), dim=0)
        result = tgt, inp
        return result

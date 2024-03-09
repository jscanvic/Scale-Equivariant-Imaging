# code adapted from
# https://raw.githubusercontent.com/ahendriksen/noise2inverse/master/noise2inverse/datasets.py

from pathlib import Path
import numpy as np
import torch
from deepinv.physics import Blur
from itertools import combinations
from torch.utils.data import (
    DataLoader,
    Dataset
)

def backproject(y, physics):
    if isinstance(physics, Blur):
        assert y.dim() == 4
        k = physics.filter.squeeze(0).squeeze(0)
        assert k.dim() == 2
        psf = torch.zeros(
            (y.shape[0], 1, y.shape[2], y.shape[3]), device=y.device, dtype=y.dtype
        )
        psf[:, 0, : k.shape[0], : k.shape[1]] = k
        psf = torch.roll(psf, (-(k.shape[0] // 2), -(k.shape[1] // 2)), dims=(2, 3))
        otf = torch.fft.rfft2(psf, dim=(2, 3))
        otf = otf.repeat(1, y.shape[1], 1, 1)

        x_hat = torch.fft.rfft2(y, dim=(2, 3))
        x_hat = x_hat / otf
        x_hat = torch.fft.irfft2(x_hat, dim=(2, 3))
        return x_hat
    else:
        return physics.A_pinv(y)

def n2i_slices(y, num_splits=4):
    slices = []

    for j in range(num_splits):
        slice = torch.zeros_like(y)
        slice[:, :, j::num_splits, :] = y[:, :, j::num_splits, :]
        slices.append(slice)

    return slices


def n2i_pair(y, physics, strategy="X:1", num_splits=4):
    slices = n2i_slices(y, num_splits=num_splits)
    slices = [backproject(slice, physics) for slice in slices]

    if strategy == "X:1":
        num_input = num_splits - 1
    else:
        num_input = 1

    split_idxs = set(range(num_splits))
    input_idxs = list(combinations(split_idxs, num_input))
    target_idxs = [split_idxs - set(idxs) for idxs in input_idxs]

    idx = np.random.randint(0, len(input_idxs))

    inputs = [slices[j] for j in input_idxs[idx]]
    targets = [slices[j] for j in target_idxs[idx]]

    inp = torch.sum(torch.stack(inputs), dim=0)
    tgt = torch.sum(torch.stack(targets), dim=0)

    return tgt, inp
# code adapted from
# https://raw.githubusercontent.com/ahendriksen/noise2inverse/master/noise2inverse/datasets.py

from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from torch.utils.data import (
    DataLoader,
    Dataset
)

def n2i_slices(y, num_splits=4):
    slices = []

    for j in range(num_splits):
        slices = torch.zeros_like(y)
        slices[:, :, j::num_splits, :] = y[:, :, j::num_splits, :]
        slices.append(slices)

    return slices


def n2i_pair(y, strategy="X:1", num_splits=4):
    slices = n2i_slices(y, num_splits=num_splits)

    if strategy == "X:1":
        num_input = self.num_splits - 1
    else:
        num_input = 1

    split_idxs = set(range(num_splits))
    input_idxs = list(combinations(split_idxs, num_input))
    target_idxs = [split_idxs - set(idxs) for idxs in self.input_idxs]

    inputs = [slices[j] for j in input_idxs]
    targets = [slices[j] for j in target_idxs]

    inp = torch.sum(torch.stack(inputs), dim=0)
    tgt = torch.sum(torch.stack(targets), dim=0)

    return inp, tgt

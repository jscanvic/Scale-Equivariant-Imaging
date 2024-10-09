# Code modified from https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html
import torch
from torch.nn import Module
import numpy as np
from deepinv.sampling import DiffPIR as DiffPIRBase
from deepinv.models import DRUNet, DiffUNet
from deepinv.optim import L2


class DiffPIR(Module):
    def __init__(self, physics, *args, **kwargs):
        super().__init__()
        self.physics = physics
        if "model" not in kwargs:
            denoiser = DRUNet(pretrained="download")
            kwargs["model"] = denoiser
        elif kwargs["model"] == "DiffUNet":
            denoiser = DiffUNet(pretrained="download")
            kwargs["model"] = denoiser
        if "data_fidelity" not in kwargs:
            kwargs["data_fidelity"] = L2()
        self.backbone = DiffPIRBase(*args, **kwargs)

    def forward(self, y):
        return self.backbone(y, self.physics)

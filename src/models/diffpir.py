# Code modified from https://deepinv.github.io/deepinv/stubs/deepinv.sampling.DiffPIR.html
import torch
from torch.nn import Module
import torch.nn.functional as F
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
            self.diffunet = True
        if "data_fidelity" not in kwargs:
            kwargs["data_fidelity"] = L2()
        self.backbone = DiffPIRBase(*args, **kwargs)

    def forward(self, y):
        if self.diffunet:
            if self.physics.task == "deblurring":
                pad_h = (32 - y.shape[2] % 32) % 32
                pad_w = (32 - y.shape[3] % 32) % 32
                y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
            else:
                pad_h = (16 - y.shape[2] % 16) % 16
                pad_w = (16 - y.shape[3] % 16) % 16
                y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")

        x_hat = self.backbone(y, self.physics)

        if self.diffunet:
            r = 1 if not hasattr(self.physics, "rate") else self.physics.rate
            out_h = r * (y.shape[2] - pad_h)
            out_w = r * (y.shape[3] - pad_w)
            x_hat = x_hat[:, :, :out_h, :out_w]

        return x_hat

import torch
from torch.nn import Module
import numpy as np
from deepinv.sampling import DPS as DPSBase
from deepinv.models import DRUNet
from deepinv.optim import L2


class DPS(Module):
    def __init__(self, physics, *args, **kwargs):
        super().__init__()
        self.physics = physics
        if "model" not in kwargs:
            denoiser = DRUNet(
                    pretrained="download",
                    device=kwargs.get("device")
                )
            kwargs["model"] = denoiser
        if "data_fidelity" not in kwargs:
            kwargs["data_fidelity"] = L2()
        self.backbone = DPSBase(*args, **kwargs)

    def forward(self, y):
        return self.backbone(y, self.physics)

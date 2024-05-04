from bm3d import bm3d_deblurring

import torch
from torch.nn import Module
import numpy as np


class BM3D(Module):
    def __init__(self, physics, sigma_psd):
        super().__init__()
        self.sigma_psd = sigma_psd
        self.kernel = physics.filter_fn

    def forward(self, y):
        psf = self.kernel.cpu().numpy()
        psf = psf[0, 0, :, :]

        x_hat = torch.empty_like(y)

        for i in range(y.shape[0]):
            for c in range(y.shape[1]):
                y_i = y[i, c, :, :].cpu().numpy()
                y_i = y_i[:, :, np.newaxis]

                sigma_psd_i = self.sigma_psd

                x_hat_i = bm3d_deblurring(y_i, sigma_psd_i, psf)
                x_hat_i = torch.from_numpy(x_hat_i).to(y.device)
                x_hat[i, c, :, :] = x_hat_i

        return x_hat

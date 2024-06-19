# Obtained from https://github.com/deepinv/deepinv/blob/9c8366497940916e70f1abb9e63c7310d85af9f7/deepinv/loss/sure.py
import torch
import torch.nn as nn
import numpy as np


def mc_div(y1, y, f, physics, tau, margin=0):
    assert margin is not None
    if margin == 0:
        b = torch.randn_like(y)
    else:
        # shape of the inner part of the measurements
        ip_shape = (y.size(0), y.size(1), y.size(2) - 2 * margin, y.size(3) - 2 * margin)
        b = torch.zeros_like(y)
        b[:, :, margin:-margin, margin:-margin] = torch.randn(*ip_shape, device=y.device, dtype=y.dtype)

    y2 = physics.A(f(y + b * tau, physics))

    out = b * (y2 - y1) / tau

    if margin != 0:
        out = out[:, :, margin:-margin, margin:-margin]

    out = out.mean()
    return out


class SureGaussianLoss(nn.Module):
    def __init__(self, sigma, tau=1e-2, margin=0, cropped_div=False, averaged_cst=False):
        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = sigma**2
        self.tau = tau
        assert margin is not None
        self.margin = margin
        self.cropped_div = cropped_div
        self.averaged_cst = averaged_cst

    def forward(self, y, x_net, physics, model, **kwargs):
        y1 = physics.A(x_net)

        if not self.cropped_div:
            div = mc_div(y1, y, model, physics, self.tau)
        else:
            div = mc_div(y1, y, model, physics, self.tau, margin=self.margin)

        div = 2 * self.sigma2 * div

        mse = y1 - y
        if self.margin != 0:
            mse = mse[:, :, self.margin:-self.margin, self.margin:-self.margin]
        mse = mse.pow(2).mean()

        if self.averaged_cst:
            loss_sure = mse + div - self.sigma2
        else:
            loss_sure = mse + div - self.sigma2 / y.size(0)

        return loss_sure

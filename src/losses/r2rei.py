import torch
from torch.nn import Module
from deepinv.loss.metric import mse

from .r2r import R2RLoss


class R2REILoss(Module):
    def __init__(self, transform, sigma, no_grad=True, metric=None):
        super().__init__()
        self.T = transform
        self.sigma = sigma
        self.no_grad = no_grad
        if metric is None:
            metric = mse()
        self.metric = metric
        self.r2r_loss = R2RLoss(eta=self.sigma, alpha=.5)

    def forward(self, *kargs, **kwargs):
        return self.r2r_loss(*kargs, **kwargs) + self.ei_loss(*kargs, **kwargs)

    # slightly modified for consistent input noise
    # base code available at https://github.com/deepinv/deepinv/blob/0b40ff5ac2f546987067465796ea55e5984d6967/deepinv/loss/ei.py
    def ei_loss(self, y, physics, model, **kwargs):
        epsilon1 = .5 * self.sigma * torch.randn_like(y)
        x1 = model(y + epsilon1, physics)

        if self.no_grad:
            with torch.no_grad():
                x2 = self.T(x1)
        else:
            x2 = self.T(x1)

        y2 = physics.A(x2)

        epsilon2 = 1.5 * self.sigma * torch.randn_like(y2)
        x3 = model(y2 + epsilon2, physics)

        return self.metric(x3, x2)

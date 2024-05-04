import torch
import torch.nn as nn


class R2RLoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss(), eta=0.1, alpha=0.5):
        super(R2RLoss, self).__init__()
        self.name = "r2r"
        self.metric = metric
        self.eta = eta
        self.alpha = alpha

    def forward(self, y, physics, model, **kwargs):
        pert = torch.randn_like(y) * self.eta

        y_plus = y + pert * self.alpha
        y_minus = y - pert / self.alpha

        output = model(y_plus, physics)

        return self.metric(physics.A(output), y_minus)

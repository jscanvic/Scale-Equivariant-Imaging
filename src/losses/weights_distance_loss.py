import torch
from torch.nn.functional import mse_loss

from copy import deepcopy

class WeightsDistanceLoss:
    def __init__(self, pretrained_model, lambd, device):
        pretrained_model = deepcopy(pretrained_model)
        pretrained_weights = pretrained_model.named_parameters()
        pretrained_weights = dict(pretrained_weights)
        self.pretrained_weights = pretrained_weights
        self.lambd = lambd
        self.device = device

    def __call__(self, model):
        keys = self.pretrained_weights.keys()
        keys = set(keys)

        weights = model.named_parameters()
        weights = dict(weights)

        assert keys == set(weights.keys())

        loss = torch.zeros((), device=self.device)
        for key in keys:
            pretrained_param = self.pretrained_weights[key]
            param = weights[key]
            loss += mse_loss(pretrained_param, param)
        return self.lambd * loss / len(keys)

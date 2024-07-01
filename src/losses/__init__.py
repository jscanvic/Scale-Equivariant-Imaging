from torch.nn import Module
from deepinv.loss import SupLoss, EILoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift

from losses.ei import Scale
from .r2r import R2REILoss
from .sure import SureGaussianLoss



class Loss(Module):
    def __init__(self, loss_fns):
        super().__init__()
        self.losses = loss_fns

    def forward(self, x, x_net, y, physics, model):
        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(x=x, x_net=x_net, y=y, physics=physics, model=model)
        return loss


def get_loss(args=args, sure_margin):
    """
    Get the losses for a given training setting

    :param str method: training method (i.e. proposed, sup, css, ei-rotate, ei-shift)
    :param float noise_level: noise level (e.g. 5)
    :param bool stop_gradient: stop the gradient for the proposed and EI methods
    """
    method = args.method
    noise_level = args.noise_level
    stop_gradient = args.stop_gradient
    sure_alternative = args.sure_alternative
    scale_antialias = args.scale_transforms_antialias
    alpha_tradeoff = args.loss_alpha_tradeoff
    sure_cropped_div = args.sure_cropped_div
    sure_averaged_cst = args.sure_averaged_cst

    assert sure_alternative in [None, "r2r"]

    if method == "proposed":
        if sure_alternative is None:
            loss_names = ["sure", "ei"]
        elif sure_alternative == "r2r":
            loss_names = ["r2rei"]
        ei_transform = Scale(antialias=scale_antialias)
    elif method == "sup":
        loss_names = ["sup"]
    elif method == "css":
        loss_names = ["sup"]
    elif method == "sure":
        loss_names = ["sure"]
    elif method == "ei-rotate":
        loss_names = ["sure", "ei"]
        ei_transform = Rotate()
    elif method == "ei-shift":
        loss_names = ["sure", "ei"]
        ei_transform = Shift()
    elif method == "noise2inverse":
        loss_names = ["sup"]

    losses = []

    for loss_name in loss_names:
        if loss_name == "sup":
            losses.append(SupLoss(metric=mse()))
        elif loss_name == "sure":
            losses.append(
                SureGaussianLoss(
                    sigma=noise_level / 255,
                    cropped_div=sure_cropped_div,
                    averaged_cst=sure_averaged_cst,
                    margin=sure_margin,
                )
            )
        elif loss_name == "r2rei":
            losses.append(
                R2REILoss(
                    transform=ei_transform,
                    sigma=noise_level / 255,
                    no_grad=stop_gradient,
                    metric=mse(),
                )
            )
        elif loss_name == "ei":
            losses.append(
                EILoss(metric=mse(),
                       transform=ei_transform,
                       no_grad=stop_gradient,
                       weight=alpha_tradeoff)
            )

    return Loss(losses)

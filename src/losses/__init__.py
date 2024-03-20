from deepinv.loss import SupLoss, SureGaussianLoss, EILoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift

from losses.ei import Scale


def get_losses(method, noise_level, stop_gradient):
    """
    Get the losses for a given training setting

    :param str method: training method (i.e. proposed, sup, css, ei-rotate, ei-shift)
    :param float noise_level: noise level (e.g. 5)
    :param bool stop_gradient: stop the gradient for the proposed and EI methods
    """
    if method == "proposed":
        loss_names = ["sure", "ei"]
        ei_transform = Scale()
    elif method == "sup":
        loss_names = ["sup"]
    elif method == "css":
        loss_names = ["sup"]
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
            losses.append(SureGaussianLoss(sigma=noise_level / 255))
        elif loss_name == "ei":
            losses.append(
                EILoss(metric=mse(), transform=ei_transform, no_grad=stop_gradient)
            )

    return losses

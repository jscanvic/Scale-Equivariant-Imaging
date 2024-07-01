from torch.nn import Module
from deepinv.loss import SupLoss, EILoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift

from losses.ei import Scale
from .r2r import R2REILoss
from .sure import SureGaussianLoss


class SupervisedLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SupLoss(metric=mse())

    def forward(self, **kwargs):
        return self.loss(**kwargs)


class CSSLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SupLoss(metric=mse())

    def forward(self, **kwargs):
        return self.loss(**kwargs)

class Noise2InverseLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SupLoss(metric=mse())

    def forward(self, **kwargs):
        return self.loss(**kwargs)

class SURELoss(Module):
    def __init__(self, noise_level, cropped_div, averaged_cst, margin):
        super().__init__()
        self.loss = SureGaussianLoss(
            sigma=noise_level / 255
            cropped_div=cropped_div,
            averaged_cst=averaged_cst,
            margin=margin,
        )

    def forward(self, **kwargs):
        return self.loss(**kwargs)

class ProposedLoss(Module):
    def __init__(self,
                 sure_alternative,
                 method,
                 scale_antialias,
                 noise_level,
                 stop_gradient,
                 sure_cropped_div,
                 sure_averaged_cst,
                 sure_margin,
                 alpha_tradeoff)
        super().__init__()

        if sure_alternative == "r2r" and method == "proposed":
            ei_transform = Scale(antialias=scale_antialias)
            loss_fns = [
                    R2REILoss(
                        transform=ei_transform,
                        sigma=noise_level / 255,
                        no_grad=stop_gradient,
                        metric=mse(),
                    )
                ]
        else:
            sure_loss = SureGaussianLoss(
                sigma=noise_level / 255,
                cropped_div=sure_cropped_div,
                averaged_cst=sure_averaged_cst,
                margin=sure_margin,
            )
            loss_fns = [sure_loss]

            if method == "proposed":
                ei_transform = Scale(antialias=scale_antialias)
            elif method == "ei-rotate":
                ei_transform = Rotate()
            elif method == "ei-shift":
                ei_transform = Shift()
            else:
                raise ValueError(f"Unknown method: {method}")

            equivariant_loss = EIloss(
                    metric=mse(),
                    transform=ei_transform,
                    no_grad=stop_gradient,
                    weight=alpha_tradeoff)

            loss_fns.append(equivariant_loss)
        self.loss_fns = loss_fns

    def forward(self, **kwargs):
        loss = 0
        for loss_fn in self.loss_fns:
            loss += loss_fn(**kwargs)
        return loss


class Loss(Module):
    def __init__(self,
                 kind,
                 sure_alternative,
                 method,
                 scale_antialias,
                 noise_level,
                 stop_gradient,
                 sure_cropped_div,
                 sure_averaged_cst,
                 sure_margin,
                 alpha_tradeoff):
        super().__init__()
        if kind == "Supervised":
            self.loss = SupervisedLoss()
        elif kind == "CSS":
            self.loss = CSSLoss()
        elif kind == "Noise2Inverse":
            self.loss = Noise2InverseLoss()
        elif kind == "SURE":
            self.loss = SURELoss(noise_level=noise_level,
                                 cropped_div=sure_cropped_div,
                                 averaged_cst=sure_averaged_cst,
                                 margin=sure_margin)
        elif kind == "Proposed":
            self.loss = ProposedLoss(sure_alternative=sure_alternative,
                                         method=method,
                                         scale_antialias=scale_antialias,
                                         noise_level=noise_level,
                                         stop_gradient=stop_gradient,
                                         sure_cropped_div=sure_cropped_div,
                                         sure_averaged_cst=sure_averaged_cst,
                                         sure_margin=sure_margin,
                                         alpha_tradeoff=alpha_tradeoff)
        else:
            raise ValueError(f"Unknown loss kind: {kind}")

    def forward(self, **kwargs):
        return self.loss(**kwargs)


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

    if method == "sup":
        kind = "Supervised"
    elif method == "css":
        kind = "CSS"
    elif method == "noise2inverse":
        kind = "Noise2Inverse"
    elif method == "sure":
        kind = "SURE"
    elif method in ["proposed", "ei-rotate", "ei-shift"]:
        kind = "Proposed"
    else:
        raise ValueError(f"Unknown method: {method}")

    loss = Loss(kind=kind,
                sure_alternative=sure_alternative,
                method=method,
                scale_antialias=scale_antialias,
                noise_level=noise_level,
                stop_gradient=stop_gradient,
                sure_cropped_div=sure_cropped_div,
                sure_averaged_cst=sure_averaged_cst,
                sure_margin=sure_margin,
                alpha_tradeoff=alpha_tradeoff)

    return loss

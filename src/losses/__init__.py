from torch.nn import Module
from deepinv.loss import SupLoss, EILoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift

from transforms import ScalingTransform
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
            sigma=noise_level / 255,
            cropped_div=cropped_div,
            averaged_cst=averaged_cst,
            margin=margin,
        )

    def forward(self, **kwargs):
        return self.loss(**kwargs)


class ProposedLoss(Module):
    def __init__(
        self,
        sure_alternative,
        method,
        scale_antialias,
        noise_level,
        stop_gradient,
        sure_cropped_div,
        sure_averaged_cst,
        sure_margin,
        alpha_tradeoff,
        transforms,
    ):
        super().__init__()

        assert sure_alternative in [None, "r2r"]
        if sure_alternative == "r2r" and method == "proposed":
            ei_transform = ScalingTransform(
                    kind="padded",
                    antialias=scale_antialias
                    )
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

            if transforms == "Scaling_Transforms":
                ei_transform = ScalingTransform(
                        kind="padded",
                        antialias=scale_antialias
                        )
            elif transforms == "Rotations":
                ei_transform = Rotate()
            elif transforms == "Shifts":
                ei_transform = Shift()
            else:
                raise ValueError(f"Unknown transforms: {transforms}")

            equivariant_loss = EILoss(
                metric=mse(),
                transform=ei_transform,
                no_grad=stop_gradient,
                weight=alpha_tradeoff,
            )

            loss_fns.append(equivariant_loss)
        self.loss_fns = loss_fns

    def forward(self, **kwargs):
        loss = 0
        for loss_fn in self.loss_fns:
            loss += loss_fn(**kwargs)
        return loss


class Loss(Module):
    def __init__(
        self,
        blueprint,
        noise_level,
        sure_cropped_div,
        sure_averaged_cst,
        sure_margin,
        method,
    ):
        super().__init__()

        if method == "supervised":
            self.loss = SupervisedLoss()
        elif method == "css":
            self.loss = CSSLoss()
        elif method == "noise2inverse":
            self.loss = Noise2InverseLoss()
        elif method == "sure":
            self.loss = SURELoss(
                noise_level=noise_level,
                cropped_div=sure_cropped_div,
                averaged_cst=sure_averaged_cst,
                margin=sure_margin,
            )
        elif method in ["proposed", "ei-rotate", "ei-shift"]:
            assert method not in [
                "ei-rotate",
                "ei-shift",
            ], f"Deprecated method: {method}"
            self.loss = ProposedLoss(
                noise_level=noise_level,
                sure_cropped_div=sure_cropped_div,
                sure_averaged_cst=sure_averaged_cst,
                sure_margin=sure_margin,
                method=method,
                **blueprint[ProposedLoss.__name__],
            )
        else:
            raise ValueError(f"Unknwon method: {method}")

    def forward(self, **kwargs):
        return self.loss(**kwargs)


def get_loss(args, sure_margin):
    blueprint = {}

    blueprint[ProposedLoss.__name__] = {
        "stop_gradient": args.ProposedLoss__stop_gradient,
        "sure_alternative": args.ProposedLoss__sure_alternative,
        "scale_antialias": args.ProposedLoss__scale_antialias,
        "alpha_tradeoff": args.ProposedLoss__alpha_tradeoff,
        "transforms": args.ProposedLoss__transforms,
    }

    method = args.method
    noise_level = args.noise_level
    sure_cropped_div = args.sure_cropped_div
    sure_averaged_cst = args.sure_averaged_cst
    sure_margin = sure_margin

    loss = Loss(
        blueprint=blueprint,
        method=method,
        noise_level=noise_level,
        sure_cropped_div=sure_cropped_div,
        sure_averaged_cst=sure_averaged_cst,
        sure_margin=sure_margin,
    )

    return loss

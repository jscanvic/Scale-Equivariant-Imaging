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

    def forward(self, x, y, physics, model):
        x_net = model(y)
        return self.loss(x=x, x_net=x_net, y=y, physics=physics, model=model)


class CSSLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SupLoss(metric=mse())

    def forward(self, x, y, physics, model):
        x_net = model(y)
        return self.loss(x=x, x_net=x_net, y=y, physics=physics, model=model)


class Noise2InverseLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = SupLoss(metric=mse())

    def forward(self, x, y, physics, model):
        x_net = model(y)
        return self.loss(x=x, x_net=x_net, y=y, physics=physics, model=model)


class SURELoss(Module):
    def __init__(self, noise_level, cropped_div, averaged_cst, margin):
        super().__init__()
        self.loss = SureGaussianLoss(
            sigma=noise_level / 255,
            cropped_div=cropped_div,
            averaged_cst=averaged_cst,
            margin=margin,
        )

    def forward(self, x, y, physics, model):
        x_net = model(y)
        return self.loss(x=x, x_net=x_net, y=y, physics=physics, model=model)


class ProposedLoss(Module):
    def __init__(
        self,
        blueprint,
        sure_alternative,
        noise_level,
        stop_gradient,
        sure_cropped_div,
        sure_averaged_cst,
        sure_margin,
        alpha_tradeoff,
        transforms,
    ):
        super().__init__()

        if transforms == "Scaling_Transforms":
            ei_transform = ScalingTransform(**blueprint[ScalingTransform.__name__])
        elif transforms == "Rotations":
            ei_transform = Rotate()
        elif transforms == "Shifts":
            ei_transform = Shift()
        else:
            raise ValueError(f"Unknown transforms: {transforms}")

        assert sure_alternative in [None, "r2r"]
        if sure_alternative == "r2r":
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

            equivariant_loss = EILoss(
                metric=mse(),
                transform=ei_transform,
                no_grad=stop_gradient,
                weight=alpha_tradeoff,
            )

            loss_fns.append(equivariant_loss)
        self.loss_fns = loss_fns

        # NOTE: This could be done better.
        if sure_alternative == "r2r":
            self.compute_x_net = False
        else:
            self.compute_x_net = True

    def forward(self, x, y, physics, model):
        if self.compute_x_net:
            x_net = model(y)
        else:
            x_net = None

        loss = 0
        for loss_fn in self.loss_fns:
            loss += loss_fn(x=x, x_net=x_net, y=y, physics=physics, model=model)
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
        elif method == "proposed":
            self.loss = ProposedLoss(
                blueprint=blueprint,
                noise_level=noise_level,
                sure_cropped_div=sure_cropped_div,
                sure_averaged_cst=sure_averaged_cst,
                sure_margin=sure_margin,
                **blueprint[ProposedLoss.__name__],
            )
        else:
            raise ValueError(f"Unknwon method: {method}")

    def forward(self, x, y, physics, model):
        return self.loss(x=x, y=y, physics=physics, model=model)


def get_loss(args, sure_margin):
    blueprint = {}

    blueprint[ProposedLoss.__name__] = {
        "stop_gradient": args.ProposedLoss__stop_gradient,
        "sure_alternative": args.ProposedLoss__sure_alternative,
        "alpha_tradeoff": args.ProposedLoss__alpha_tradeoff,
        "transforms": args.ProposedLoss__transforms,
    }

    blueprint[ScalingTransform.__name__] = {
        "kind": args.ScalingTransform__kind,
        "antialias": args.ScalingTransform__antialias,
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

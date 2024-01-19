import torch
import torch.nn.functional as F
from deepinv.loss import SupLoss, SureGaussianLoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift
from torch.nn import Module, Parameter


class EILoss(Module):
    r"""
    Equivariant imaging self-supervised loss.

    Assumes that the set of signals is invariant to a group of transformations (rotations, translations, etc.)
    in order to learn from incomplete measurement data alone https://https://arxiv.org/pdf/2103.14756.pdf.

    The EI loss is defined as

    .. math::

        \| T_g \hat{x} - \inverse{\forw{T_g \hat{x}}}\|^2


    where :math:`\hat{x}=\inverse{y}` is a reconstructed signal and
    :math:`T_g` is a transformation sampled at random from a group :math:`g\sim\group`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param deepinv.Transform, torchvision.transforms transform: Transform to generate the virtually
        augmented measurement. It can be any torch-differentiable function (e.g., a ``torch.nn.Module``).
    :param torch.nn.Module metric: Metric used to compute the error between the reconstructed augmented measurement and the reference
        image.
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    :param float weight: Weight of the loss.
    :param bool no_grad: if ``True``, the gradient does not propagate through :math:`T_g`. Default: ``True``.
    """

    def __init__(
        self,
        transform,
        metric=torch.nn.MSELoss(),
        apply_noise=True,
        weight=1.0,
        no_grad=True,
        byol_params=None,
    ):
        super(EILoss, self).__init__()
        self.name = "ei"
        self.metric = metric
        self.weight = weight
        self.T = transform
        self.noise = apply_noise
        self.no_grad = no_grad
        self.byol_params = byol_params

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the EI loss

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """

        if self.byol_params is not None:
            # set model params as byol params temporarily
            original_params = {
                k: Parameter(param.data, requires_grad=param.requires_grad) for k, param in model.named_parameters()
            }
            for name, param in model.named_parameters():
                param.data = self.byol_params[name].data

            x_net = model(y, physics)

            # restore original params
            for name, param in model.named_parameters():
                param.data = original_params[name].data

        if self.no_grad:
            with torch.no_grad():
                x2 = self.T(x_net)
        else:
            x2 = self.T(x_net)

        if self.noise:
            y2 = physics(x2)
        else:
            y2 = physics.A(x2)

        x3 = model(y, physics)

        loss_ei = self.weight * self.metric(x3, x2)
        return loss_ei


def sample_from(values, shape=(1,), dtype=torch.float32, device="cpu"):
    """Sample a random tensor from a list of values"""
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = torch.floor(N * torch.rand(shape, device=device, dtype=dtype)).to(
        torch.int
    )
    return values[indices]


class Scale(Module):
    """
    2D Scaling.

    Resample the input image on a grid obtained using
    an isotropic dilation, with random scale factor
    and origin. By default, the input image is viewed
    as periodic and the output image is effectively padded
    by reflections. Additionally, resampling is performed
    using bicubic interpolation.

    :param list factors: list of scale factors (default: [.75, .5])
    :param str padding_mode: padding mode for grid sampling
    :param str mode: interpolation mode for grid sampling
    """

    def __init__(self, factors=None, padding_mode="reflection", mode="bicubic"):
        super().__init__()

        self.factors = factors or [0.75, 0.5]
        self.padding_mode = padding_mode
        self.mode = mode

    def forward(self, x):
        b, _, h, w = x.shape

        # Sample a random scale factor for each batch element
        factor = sample_from(self.factors, shape=(b,), device=x.device)
        factor = factor.view(b, 1, 1, 1).repeat(1, 1, 1, 2)

        # Sample a random transformation center for each batch element
        # with coordinates in [-1, 1]
        center = torch.rand((b, 2), dtype=x.dtype, device=x.device)
        center = center.view(b, 1, 1, 2)
        center = 2 * center - 1

        # Compute the sampling grid for the scale transformation
        u = torch.arange(w, dtype=x.dtype, device=x.device)
        v = torch.arange(h, dtype=x.dtype, device=x.device)
        u = 2 / w * u - 1
        v = 2 / h * v - 1
        U, V = torch.meshgrid(u, v)
        grid = torch.stack([V, U], dim=-1)
        grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
        grid = 1 / factor * (grid - center) + center

        return F.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=True
        )


def get_losses(method, noise_level, stop_gradient, byol_params=None):
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

    losses = []

    for loss_name in loss_names:
        if loss_name == "sup":
            losses.append(SupLoss(metric=mse()))
        elif loss_name == "sure":
            losses.append(SureGaussianLoss(sigma=noise_level / 255))
        elif loss_name == "ei":
            losses.append(
                EILoss(
                    metric=mse(),
                    transform=ei_transform,
                    no_grad=stop_gradient,
                    byol_params=byol_params,
                )
            )

    return losses

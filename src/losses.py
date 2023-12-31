import torch
import torch.nn.functional as F
from deepinv.loss import SupLoss, SureGaussianLoss, EILoss
from deepinv.loss.metric import mse
from deepinv.transform import Rotate, Shift
from torch.nn import Module


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

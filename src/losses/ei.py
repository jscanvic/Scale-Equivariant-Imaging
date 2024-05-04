import torch
from torch.nn import Module, functional as F


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

    def __init__(
        self, factors=None, padding_mode="reflection", mode="bicubic", antialias=False
    ):
        super().__init__()

        self.factors = factors or [0.75, 0.5]
        self.padding_mode = padding_mode
        self.mode = mode
        self.antialias = antialias

    def forward(self, x):
        b, _, h, w = x.shape

        # Sample a random scale factor for each batch element
        factor = sample_from(self.factors, shape=(b,), device=x.device)

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
        grid = 1 / factor.view(b, 1, 1, 1).expand_as(grid) * (grid - center) + center

        if self.antialias:
            xs = []
            for i in range(x.shape[0]):
                # Apply the anti-aliasing filter
                z = F.interpolate(
                    x[i : i + 1, :, :, :],
                    scale_factor=factor[i].item(),
                    mode=self.mode,
                    antialias=True,
                )
                z = F.grid_sample(
                    z,
                    grid[i : i + 1],
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=True,
                )
                z = z.squeeze(0)
                xs.append(z)
            x = torch.stack(xs)
        else:
            x = F.grid_sample(
                x,
                grid,
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=True,
            )

        return x

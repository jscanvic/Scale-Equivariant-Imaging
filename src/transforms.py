import torch
from torch.nn import Module, functional as F


def sample_from(values, shape=(1,), dtype=torch.float32, device="cpu"):
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = torch.floor(N * torch.rand(shape, device=device, dtype=dtype)).to(
        torch.int
    )
    return values[indices]


def sample_scaling_parameters(image_count, device, dtype):
    # Sample a random scale factor for each batch element
    factor = sample_from([0.75, 0.5], shape=(image_count,), device=device)

    # Sample a random transformation center for each batch element
    # with coordinates in [-1, 1]
    center = torch.rand((image_count, 2), dtype=dtype, device=device)
    center = center.view(image_count, 1, 1, 2)
    center = 2 * center - 1

    return factor, center


def padded_scaling_transform(x, factor, center, mode, padding_mode, antialiased):
    b, _, h, w = x.shape

    # Compute the sampling grid for the scale transformation
    u = torch.arange(w, dtype=x.dtype, device=x.device)
    v = torch.arange(h, dtype=x.dtype, device=x.device)
    u = 2 / w * u - 1
    v = 2 / h * v - 1
    U, V = torch.meshgrid(u, v)
    grid = torch.stack([V, U], dim=-1)
    grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
    grid = 1 / factor.view(b, 1, 1, 1).expand_as(grid) * (grid - center) + center

    if not antialiased:
        x = F.grid_sample(
            x,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )
    else:
        xs = []
        for i in range(x.shape[0]):
            # Apply the anti-aliasing filter
            z = F.interpolate(
                x[i : i + 1, :, :, :],
                scale_factor=factor[i].item(),
                mode=mode,
                antialias=True,
            )
            z = F.grid_sample(
                z,
                grid[i : i + 1],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=True,
            )
            z = z.squeeze(0)
            xs.append(z)
        x = torch.stack(xs)
    return x


class PaddedScalingTransform(Module):
    def __init__(self, antialias=False):
        super().__init__()
        self.antialias = antialias

    def forward(self, x):
        factor, center = sample_scaling_parameters(
                image_count=x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )

        x = padded_scaling_transform(
            x,
            factor=factor,
            center=center,
            antialiased=self.antialias,
            mode="bicubic",
            padding_mode="reflection",
        )

        return x


class ScalingTransform(Module):
    def __init__(self, antialias=False):
        super().__init__()
        self.transform = PaddedScalingTransform(antialias=antialias)

    def forward(self, x):
        return self.transform(x)

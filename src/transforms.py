import torch
from torch.nn import Module, functional as F


def sample_from(values, shape=(1,), dtype=torch.float32, device="cpu"):
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = torch.floor(N * torch.rand(shape, device=device, dtype=dtype)).to(
        torch.int
    )
    return values[indices]


def sample_downsampling_parameters(image_count, device, dtype):
    downsampling_rate = sample_from([0.75, 0.5], shape=(image_count,), dtype=dtype, device=device)

    # The coordinates are in [-1, 1].
    center = torch.rand((image_count, 2), dtype=dtype, device=device)
    center = center.view(image_count, 1, 1, 2)
    center = 2 * center - 1

    return downsampling_rate, center


def get_downsampling_grid(shape, downsampling_rate, center, dtype, device):
    b, _, h, w = shape

    # Compute the sampling grid for the scale transformation
    u = torch.arange(w, dtype=dtype, device=device)
    v = torch.arange(h, dtype=dtype, device=device)
    u = 2 / w * u - 1
    v = 2 / h * v - 1
    U, V = torch.meshgrid(u, v)
    grid = torch.stack([V, U], dim=-1)
    grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
    grid = 1 / downsampling_rate.view(b, 1, 1, 1).expand_as(grid) * (grid - center) + center

    return grid


def alias_free_interpolate(x, downsampling_rate, interpolation_mode):
    xs = []
    for i in range(x.shape[0]):
        z = F.interpolate(
            x[i : i + 1, :, :, :],
            scale_factor=downsampling_rate[i].item(),
            mode=interpolation_mode,
            antialias=True,
        )
        z = z.squeeze(0)
        xs.append(z)
    return torch.stack(xs)


def padded_downsampling_transform(x, downsampling_rate, center, mode, padding_mode, antialiased):
    shape = x.shape

    if antialiased:
        x = alias_free_interpolate(x, downsampling_rate=downsampling_rate, interpolation_mode=mode)

    grid = get_downsampling_grid(shape=shape, downsampling_rate=downsampling_rate, center=center, dtype=x.dtype, device=x.device)
    return F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


class PaddedDownsamplingTransform(Module):
    def __init__(self, antialias=False):
        super().__init__()
        self.antialias = antialias

    def forward(self, x):
        downsampling_rate, center = sample_downsampling_parameters(
                image_count=x.shape[0],
                device=x.device,
                dtype=x.dtype,
            )

        x = padded_downsampling_transform(
            x,
            downsampling_rate=downsampling_rate,
            center=center,
            antialiased=self.antialias,
            mode="bicubic",
            padding_mode="reflection",
        )

        return x


def downsampling_transform(x, downsampling_rate, mode, antialiased):
    xs = []
    for i in range(x.shape[0]):
        z = F.interpolate(
            x[i : i + 1, :, :, :],
            scale_factor=downsampling_rate[i].item(),
            mode=mode,
            antialias=antialiased,
        )
        z = z.squeeze(0)
        xs.append(z)
    return torch.stack(xs)


class NormalDownsamplingTransform(Module):
    def __init__(self, antialias=False):
        super().__init__()
        self.antialias = antialias

    def forward(self, x):
        downsampling_rate, _ = sample_downsampling_parameters(
            image_count=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        x = downsampling_transform(
            x,
            downsampling_rate=downsampling_rate,
            mode="bicubic",
            antialiased=self.antialias,
        )

        return x


class ScalingTransform(Module):
    def __init__(self, kind, antialias):
        super().__init__()
        if kind == "padded":
            self.transform = PaddedDownsamplingTransform(antialias=antialias)
        elif kind == "normal":
            self.transform = NormalDownsamplingTransform(antialias=antialias)
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def forward(self, x):
        return self.transform(x)

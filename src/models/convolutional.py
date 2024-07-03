import torch
from torch.nn import Module, Sequential, ModuleList
from torch.nn.parameter import Parameter

from physics import Blur

import torch
from torch.nn import (
    Module,
    Sequential,
    ModuleList,
    Conv2d,
    GELU,
    LayerNorm as BaseLayerNorm,
)
import torch.nn.functional as F
from math import ceil


# a layer norm averaging over the channels
class LayerNorm(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = BaseLayerNorm(*args, **kwargs)

    def forward(self, x):
        x = torch.swapaxes(x, -3, -1)
        x = self.ln(x)
        x = torch.swapaxes(x, -3, -1)
        return x


class ConvBlock(Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim
        )
        self.ln = LayerNorm(dim, eps=1e-6)
        self.conv2 = Conv2d(in_channels=dim, out_channels=4 * dim, kernel_size=1)
        self.gelu = GELU()
        self.conv3 = Conv2d(in_channels=4 * dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.ln(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu(x1)
        x1 = self.conv3(x1)
        return x + x1


class IdealUpsample(Module):
    def __init__(self, rate=2):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        s = (x.shape[-2], x.shape[-1])
        x = torch.fft.rfft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))

        x2 = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] * self.rate, x.shape[3] * self.rate),
            device=x.device,
            dtype=x.dtype,
        )

        margin_v = (x.shape[-2] * (self.rate - 1)) // 2
        margin_h = (x.shape[-1] * (self.rate - 1)) // 2
        if x.shape[-2] % 2 == 1:
            margin_t = margin_v + 1
            margin_b = margin_v
        else:
            margin_t = margin_v
            margin_b = margin_v

        if x.shape[-1] % 2 == 1:
            margin_l = margin_h + 1
            margin_r = margin_h
        else:
            margin_l = margin_h
            margin_r = margin_h

        x2[:, :, margin_t:-margin_b, margin_l:-margin_r] = x
        x = x2

        torch.fft.ifftshift(x, dim=(-2, -1))
        s = (s[0] * self.rate, s[1] * self.rate)
        x = torch.fft.irfft2(x, dim=(-2, -1), s=s)
        return x


class Upsample(Module):
    def __init__(self, in_channels, out_channels=None, rate=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels // (rate**2)
        self.rate = rate

        self.seq = Sequential()
        self.seq.append(
                IdealUpsample(rate=self.rate)
            )
        self.seq.append(
                LayerNorm(self.in_channels, eps=1e-6)
            )
        self.seq.append(
                Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)
            )

    def forward(self, x):
        return self.seq(x)



class IdealDownsample(Module):
    def __init__(self, rate=2):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        s = (x.shape[-2], x.shape[-1])
        x = torch.fft.rfft2(x, dim=(-2, -1))
        x = torch.fft.fftshift(x, dim=(-2, -1))

        # half crop size with respect to height (resp. width)
        hcsh = ceil(x.shape[-2] / (2 * self.rate))
        hcsw = ceil(x.shape[-1] / (2 * self.rate))

        otf = torch.zeros_like(x)
        otf[:, :, hcsh:-hcsh, hcsw:-hcsw] = 1
        x = otf * x

        torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.irfft2(x, dim=(-2, -1), s=s)
        return x[:, :, :: self.rate, :: self.rate]


class Downsample(Module):
    def __init__(self, in_channels, out_channels=None, rate=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels * (rate**2)
        self.rate = rate
        self.ln = LayerNorm(self.in_channels, eps=1e-6)
        self.conv = Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, stride=1
        )
        self.ideal_downsample = IdealDownsample(rate=self.rate)

    def forward(self, x):
        x = self.ln(x)
        x = self.conv(x)
        x = self.ideal_downsample(x)
        return x

class UNet(Module):
    def __init__(
        self,
        in_channels,
        scales=5,
        num_conv_blocks=5,
        rate=2,
        residual=False,
    ):
        super().__init__()
        self.scales = scales
        self.residual = residual

        self.conv_sequences = ModuleList()
        self.downsampling_layers = ModuleList()
        self.upsampling_layers = ModuleList()

        layer_in_channels = in_channels
        for _ in range(scales - 1):
            # append the conv blocks
            conv_sequence = Sequential()
            for _ in range(num_conv_blocks):
                layer = ConvBlock(dim=layer_in_channels)
                conv_sequence.append(layer)
            self.conv_sequences.append(conv_sequence)

            # append the downsampling layer
            layer = Downsample(in_channels=layer_in_channels)
            self.downsampling_layers.append(layer)
            layer_in_channels = layer_in_channels * pow(rate, 2)

        # append the conv blocks
        conv_sequence = Sequential()
        for _ in range(num_conv_blocks):
            layer = ConvBlock(dim=layer_in_channels)
            conv_sequence.append(layer)
        self.conv_sequences.append(conv_sequence)

        for _ in range(scales - 1):
            # append the upsampling layer
            layer = Upsample(in_channels=layer_in_channels, rate=rate)
            self.upsampling_layers.append(layer)
            layer_in_channels = layer_in_channels // pow(rate, 2)

            conv_sequence = Sequential()
            for _ in range(num_conv_blocks):
                layer = ConvBlock(dim=layer_in_channels)
                conv_sequence.append(layer)
            self.conv_sequences.append(conv_sequence)

    def forward(self, x):
        x0 = x

        conv_it = iter(self.conv_sequences)
        downsampling_it = iter(self.downsampling_layers)
        upsampling_it = iter(self.upsampling_layers)

        queue = []

        for _ in range(self.scales - 1):
            x = next(conv_it)(x)
            queue.append(x)
            x = next(downsampling_it)(x)

        x = next(conv_it)(x)

        for _ in range(self.scales - 1):
            x = next(upsampling_it)(x)
            # skip connection
            x2 = queue.pop()
            x = x + x2
            x = next(conv_it)(x)

        if self.residual:
            x = x + x0

        return x


class ConvNeuralNetwork(Module):
    def __init__(self, in_channels, upsampling_rate, unet_residual, num_conv_blocks=5, scales=5):
        super().__init__()
        self.seq = Sequential()
        self.scales = scales

        if upsampling_rate != 1:
            module = Upsample(in_channels=in_channels, out_channels=in_channels, rate=upsampling_rate)
            self.seq.append(module)

        module = UNet(
            in_channels=in_channels,
            scales=scales,
            num_conv_blocks=num_conv_blocks,
            residual=unet_residual,
        )
        self.seq.append(module)

    def forward(self, y):
        div = 2 ** (self.scales - 1)
        pad_h = (div - y.shape[-2] % div) % div
        pad_w = (div - y.shape[-1] % div) % div

        if pad_h != 0 or pad_w != 0:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")

        x_hat = self.seq(y)

        if pad_h != 0 and pad_w != 0:
            x_hat = x_hat[:, :, :-pad_h, :-pad_w]
        elif pad_h != 0:
            x_hat = x_hat[:, :, :-pad_h, :]
        elif pad_w != 0:
            x_hat = x_hat[:, :, :, :-pad_w]

        return x_hat

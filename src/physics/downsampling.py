import torch
from deepinv.physics import LinearPhysics
from torch.nn import Module
from torch.nn.functional import interpolate
from torch.signal.windows import kaiser

from imresize import imresize


class KaiserFilter(Module):
    def __init__(self, antialias=True, window_length=5, beta=12.0):
        super().__init__()
        self.antialias = antialias
        self.window_length = window_length
        self.beta = beta

        kernel = kaiser(self.window_length, beta=self.beta, sym=True)
        self.kernel = kernel / kernel.sum()

    def forward(self, x):
        kernel = self.kernel.to(x.device, x.dtype)
        x = self.filter1d(x, dim=2, kernel=kernel)
        x = self.filter1d(x, dim=3, kernel=kernel)
        return x

    @staticmethod
    def filter1d(x, dim, kernel):
        # this makes it easier to broadcast the otf
        x = x.swapaxes(dim, -1)

        psf = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        psf[: kernel.shape[-1]] = kernel
        psf = torch.roll(psf, -(kernel.shape[-1] // 2), dims=-1)

        n = x.shape[-1]
        x = torch.fft.rfft(x, dim=-1)
        otf = torch.fft.rfft(psf, dim=-1)
        otf = otf.expand(x.shape)
        x = x * otf
        x = torch.fft.irfft(x, dim=-1, n=n)

        # restore axes order
        x = x.swapaxes(dim, -1)

        return x


class Downsampling(LinearPhysics):
    """
    Downsampling operator built upon Matlab's imresize function

    :param int factor: downsampling factor
    :param bool antialias: antialiasing
    """

    def __init__(self, factor=2, antialias=True, filter="bicubic"):
        super().__init__()
        self.factor = factor
        self.antialias = antialias
        assert filter in ["bicubic", "bicubic_torch", "kaiser"]
        self.filter = filter
        if self.filter == "bicubic":
            self.filter_fn = None
        elif self.filter == "bicubic_torch":
            self.filter_fn = None
        elif self.filter == "kaiser":
            self.filter_fn = KaiserFilter(antialias=self.antialias)

    def A(self, x):
        if self.filter == "bicubic":
            y = imresize(x, scale=1 / self.factor, antialiasing=self.antialias)
        elif self.filter == "bicubic_torch":
            y = interpolate(x, scale_factor=1 / self.factor, mode="bicubic")
        elif self.filter == "kaiser":
            y = self.filter_fn(x)
            y = y[:, :, :: self.factor, :: self.factor]
        return y

    def A_adjoint(self, y):
        if self.filter == "bicubic":
            x = imresize(y, scale=self.factor)
        elif self.filter == "bicubic_torch":
            x = interpolate(y, scale_factor=self.factor, mode="bicubic")
        elif self.filter == "kaiser":
            x = interpolate(y, scale_factor=self.factor, mode="bicubic")
        return x

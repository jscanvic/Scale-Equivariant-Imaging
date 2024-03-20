import torch
from deepinv.physics import LinearPhysics


class CTLikeFilter(LinearPhysics):
    def __init__(self, eps=1):
        super().__init__()
        self.eps = eps

    def A(self, x):
        x = self.filter1d(x, dim=2, inverse=True)
        x = self.filter1d(x, dim=3, inverse=True)
        return x

    def A_dagger(self, y):
        y = self.filter1d(y, dim=2, inverse=False)
        y = self.filter1d(y, dim=3, inverse=False)
        return y

    def filter1d(self, x, dim, inverse=False):
        # we swap axes to make the last axis the one to be filtered
        # making the code easier to understand
        x = x.swapaxes(dim, -1)

        n = x.shape[-1]
        x = torch.fft.rfft(x, dim=-1)
        otf = torch.arange(x.shape[-1], device=x.device).to(dtype=x.dtype)
        otf = otf + self.eps
        if inverse:
            otf = 1 / otf

        otf = otf.expand(x.shape)
        x = x * otf
        x = torch.fft.irfft(x, dim=-1, n=n)

        # restore axes order
        x = x.swapaxes(dim, -1)

        return x

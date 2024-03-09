import torch
from deepinv.physics import Blur, GaussianNoise, LinearPhysics
from torch.signal.windows import kaiser

from imresize import imresize


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
        self.filter = filter

    def A(self, x):
        if self.filter == "bicubic":
            x2 = imresize(x, scale=1 / self.factor, antialiasing=self.antialias)
        elif self.filter == "kaiser":
            window_length = 5
            beta = 12.0
            kernel_1d = kaiser(window_length, beta=beta, sym=True, dtype=x.dtype, device=x.device)
            kernel_1d = kernel_1d / kernel_1d.sum()

            psf_h = torch.zeros(x.shape[0], 1, x.shape[2], 1, dtype=x.dtype, device=x.device)
            psf_h[:, 0, :kernel_1d.shape[0], 0] = kernel_1d
            psf_h = torch.roll(psf_h, -(kernel_1d.shape[0] // 2), dims=2)
            otf_h = torch.fft.fft(psf_h, dim=2)

            psf_w = torch.zeros(x.shape[0], 1, 1, x.shape[3], dtype=x.dtype, device=x.device)
            psf_w[:, 0, 0, :kernel_1d.shape[0]] = kernel_1d
            psf_w = torch.roll(psf_w, -(kernel_1d.shape[0] // 2), dims=3)
            otf_w = torch.fft.rfft(psf_w, dim=3)

            otf = otf_h * otf_w

            # apply the Kaizer anti-aliasing filter
            x2 = torch.fft.rfft2(x, dim=(2, 3))
            x2 = otf * x2
            x2 = torch.fft.irfft2(x2, dim=(2, 3))

            # subsample
            x2 = x2[:, :, ::self.factor, ::self.factor]

        return x2

    def A_adjoint(self, x):
        if self.filter == "bicubic":
            x2 = imresize(x, scale=self.factor, antialiasing=self.antialias)
        return x2


def get_physics(task, noise_level, kernel_path=None, sr_factor=None, device="cpu", sr_filter="bicubic"):
    """
    Get the forward model for the given task

    :param task: task to perform (i.e. sr or denoising)
    :param noise_level: noise level (e.g. 5)
    :param kernel_path: path to the blur kernel (optional)
    :param sr_factor: super-resolution factor (optional)
    :param device: device to use
    """
    assert task in ["deblurring", "sr"]

    if task == "deblurring":
        kernel = torch.load(kernel_path)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
        physics = Blur(filter=kernel, padding="circular", device=device)
    else:
        physics = Downsampling(sr_factor, antialias=True, filter=sr_filter)

    physics.noise_model = GaussianNoise(sigma=noise_level / 255)

    return physics

import torch
from deepinv.physics import Blur, GaussianNoise, LinearPhysics

from imresize import imresize


class Downsampling(LinearPhysics):
    """
    Downsampling operator built upon Matlab's imresize function

    :param int factor: downsampling factor
    :param bool antialias: antialiasing
    """
    def __init__(self, factor=2, antialias=True):
        super().__init__()
        self.factor = factor
        self.antialias = antialias

    def A(self, x):
        return imresize(x, scale=1 / self.factor, antialiasing=self.antialias)


def get_physics(task, noise_level, kernel_path=None, sr_factor=None, device="cpu"):
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
        physics = Downsampling(sr_factor, antialias=True)

    physics.noise_model = GaussianNoise(sigma=noise_level / 255)

    return physics

import torch
from deepinv.physics import GaussianNoise
from os.path import exists

from .ct_like_filter import CTLikeFilter
from .downsampling import Downsampling
from .kernels import get_kernel
from .blur import Blur

# NOTE: There should ideally be a class combining the blur and downsampling operators.

# NOTE: The borders of blurred out images should be cropped out in order to avoid boundary effects.

def get_physics(
    task,
    noise_level,
    kernel_path=None,
    sr_factor=None,
    device="cpu",
    sr_filter="bicubic",
):
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
        if kernel_path != "ct_like":
            if exists(kernel_path):
                kernel = torch.load(kernel_path)
            else:
                kernel = get_kernel(name=kernel_path)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
            physics = Blur(filter=kernel, padding="circular", device=device)
        else:
            physics = CTLikeFilter()
    else:
        physics = Downsampling(sr_factor, antialias=True, filter=sr_filter)

    physics.noise_model = GaussianNoise(sigma=noise_level / 255)

    return physics

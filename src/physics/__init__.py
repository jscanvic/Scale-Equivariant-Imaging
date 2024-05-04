import torch
from deepinv.physics import Blur, GaussianNoise

from .ct_like_filter import CTLikeFilter
from .downsampling import Downsampling


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
            kernel = torch.load(kernel_path)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
            physics = Blur(filter=kernel, padding="circular", device=device)
        else:
            physics = CTLikeFilter()
    else:
        physics = Downsampling(sr_factor, antialias=True, filter=sr_filter)

    physics.noise_model = GaussianNoise(sigma=noise_level / 255)

    return physics

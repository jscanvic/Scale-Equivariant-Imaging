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
    true_adjoint=False,
):
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
        physics = Downsampling(ratio=sr_factor, antialias=True, true_adjoint=true_adjoint)

    physics.noise_model = GaussianNoise(sigma=noise_level / 255)

    return physics

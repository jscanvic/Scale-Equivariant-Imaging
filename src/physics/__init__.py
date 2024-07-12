import torch
from deepinv.physics import GaussianNoise
from os.path import exists

from rng import fork_rng
from .ct_like_filter import CTLikeFilter
from .downsampling import Downsampling
from .kernels import get_kernel
from .blur import Blur, BlurV2

# NOTE: There should ideally be a class combining the blur and downsampling operators.


# NOTE: This might be better as a subclass of torch.Tensor. (See
# torch.nn.Parameter for reference.)
class BlurKernel:
    def __init__(self, kernel_path):
        self.kernel_path = kernel_path

    def to_tensor(self, device):
        if exists(self.kernel_path):
            kernel = torch.load(self.kernel_path)
        else:
            kernel = get_kernel(name=self.kernel_path)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
        return kernel


class PhysicsManager:

    def __init__(
        self,
        blueprint,
        task,
        device,
        noise_level,
        v2,
    ):
        if task == "deblurring":
            blur_kernel = BlurKernel(**blueprint[BlurKernel.__name__])
            kernel = blur_kernel.to_tensor(device)
            if v2:
                physics = BlurV2(kernel=kernel)
            else:
                physics = Blur(filter=kernel, padding="circular", device=device)
        elif task == "sr":
            physics = Downsampling(antialias=True, **blueprint[Downsampling.__name__])
        elif task == "invert_a_tomography_like_filter":
            physics = CTLikeFilter()
        else:
            raise ValueError(f"Unknown task: {task}")

        physics.noise_model = GaussianNoise(sigma=noise_level / 255)

        # NOTE: These are meant to go.
        setattr(self, "task", task)
        setattr(physics, "task", task)
        setattr(physics, "__manager", self)

        self.physics = physics

    def get_physics(self):
        return self.physics

    def randomly_degrade(self, x, seed):
        # NOTE: Forking the RNG and setting the seed could be done all at once.
        preserve_rng_state = seed is not None
        with fork_rng(enabled=preserve_rng_state):
            if seed is not None:
                torch.manual_seed(seed)

            x = self.physics.A(x)
            x = self.physics.noise_model(x)
        return x


# NOTE: The borders of blurred out images should be cropped out in order to avoid boundary effects.


def get_physics(args, device):
    blueprint = {}
    blueprint[PhysicsManager.__name__] = {
        "task": args.task,
        "noise_level": args.noise_level,
        "v2": args.physics_v2,
    }

    blueprint[BlurKernel.__name__] = {
        "kernel_path": args.kernel,
    }

    blueprint[Downsampling.__name__] = {
        "rate": args.sr_factor,
        "true_adjoint": args.physics_true_adjoint,
    }

    physics_manager = PhysicsManager(
        blueprint=blueprint,
        device=device,
        **blueprint[PhysicsManager.__name__],
    )
    return physics_manager.get_physics()

import torch
from deepinv.physics import GaussianNoise
from os.path import exists

from rng import fork_rng
from .ct_like_filter import CTLikeFilter
from .downsampling import Downsampling
from .kernels import get_kernel
from .blur import Blur

# NOTE: There should ideally be a class combining the blur and downsampling operators.


class PhysicsManager:

    def __init__(
        self,
        task,
        noise_level,
        kernel_path=None,
        sr_factor=None,
        device="cpu",
        true_adjoint=False,
    ):
        self.task = task
        self.noise_level = noise_level
        self.kernel_path = kernel_path
        self.sr_factor = sr_factor
        self.device = device
        self.true_adjoint = true_adjoint

        if self.task == "deblurring":
            if self.kernel_path != "ct_like":
                if exists(self.kernel_path):
                    kernel = torch.load(self.kernel_path)
                else:
                    kernel = get_kernel(name=self.kernel_path)
                kernel = kernel.unsqueeze(0).unsqueeze(0).to(self.device)
                self.physics = Blur(
                    filter=kernel, padding="circular", device=self.device
                )
            else:
                self.physics = CTLikeFilter()
        elif self.task == "sr":
            self.physics = Downsampling(
                ratio=self.sr_factor, antialias=True, true_adjoint=self.true_adjoint
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        # NOTE: These are meant to go.
        setattr(self.physics, "task", self.task)
        setattr(self.physics, "__manager", self)

        self.physics.noise_model = GaussianNoise(sigma=self.noise_level / 255)

    def get_physics(self):
        return self.physics

    def randomly_degrade(self, x, seed):
        preserve_rng_state = seed is not None
        with fork_rng(enabled=preserve_rng_state):
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
        "kernel_path": args.kernel,
        "sr_factor": args.sr_factor,
        "true_adjoint": args.physics_true_adjoint,
    }

    physics_manager = PhysicsManager(
        device=device,
        **blueprint[PhysicsManager.__name__],
    )
    return physics_manager.get_physics()

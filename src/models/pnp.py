# code from https://deepinv.github.io/deepinv/auto_examples/plug-and-play/demo_vanilla_PnP.html

from torch.nn import Module
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.dpir import get_DPIR_params


class PnPModel(Module):
    def __init__(
        self,
        physics,
        noise_level_img,
        early_stop=False,
        channels=3,
        device="cpu",
    ):
        super().__init__()

        self.physics = physics

        if noise_level_img == 0:
            noise_level_img = 1e-5

        sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)

        params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser}

        data_fidelity = L2()

        denoiser = DRUNet(
            in_channels=channels,
            out_channels=channels,
            pretrained="download",
            device=device,
        )

        prior = PnP(denoiser=denoiser)

        self.model = optim_builder(
            iteration="HQS",
            prior=prior,
            data_fidelity=data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            params_algo=params_algo,
            verbose=True,
        )

    def forward(self, y):
        return self.model(y, self.physics)

from deepinv.models import SwinIR
from torch import nn
from torch.nn.parallel import DataParallel
from torch.nn import Module

from .pnp import PnPModel
from .dip import DeepImagePrior
from .bm3d_deblurring import BM3D
from .upsample import Upsample
from .diffpir import DiffPIR
from .dps import DPS
# from .tv import TV


class Identity(Module):
    def forward(self, y):
        return y


def get_model(
    task,
    sr_factor=None,
    noise_level=None,
    physics=None,
    channels=3,
    device="cpu",
    kind="swinir",
    data_parallel_devices=None,
    dip_iterations=4000,
):
    """
    Get a model with randomly initialized weights for the given task

    :param task: task to perform (i.e. sr or denoising)
    :param sr_factor: super-resolution factor (optional)
    """
    if kind == "swinir":
        upscale = sr_factor if task == "sr" else 1
        upsampler = "pixelshuffle" if task == "sr" else None
        model = SwinIR(
            img_size=48,
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            window_size=8,
            mlp_ratio=2,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upscale=upscale,
            img_range=1.0,
            upsampler=upsampler,
            resi_connection="1conv",
            pretrained=None,
        )
    elif kind == "dip":
        model = DeepImagePrior(
            physics=physics, sr_factor=sr_factor, iterations=dip_iterations
        )
    elif kind == "pnp":
        noise_level_img = noise_level / 255
        early_stop = True

        model = PnPModel(
            physics,
            noise_level_img,
            early_stop=early_stop,
            device=device,
            channels=channels,
        )
    elif kind == "bm3d":
        model = BM3D(physics=physics, sigma_psd=noise_level / 255)
    elif kind == "diffpir":
        model = DiffPIR(physics=physics)
    elif kind == "dps":
        model = DPS(physics=physics, device=device)
    elif kind == "tv":
        model = TV(physics=physics)
    elif kind == "id":
        model = Identity()
    elif kind == "up":
        model = Upsample(factor=sr_factor)
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if data_parallel_devices is not None:
        devices = data_parallel_devices
        model = DataParallel(model, device_ids=devices, output_device=device)

    return model

from deepinv.models import SwinIR
from torch import nn
from torch.nn import Module
from dip import DIPModel
from pnp import PnPModel
from bm3d_model import BM3DModel
from imresize import imresize


class Identity(Module):
    def forward(self, y):
        return y

class Upsample(Module):
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def forward(self, y):
        return imresize(y, scale=self.factor)


def get_model(task, sr_factor=None, noise_level=None, physics=None, channels=3, device="cpu", kind="swinir"):
    """
    Get a model with randomly initialized weights for the given task

    :param task: task to perform (i.e. sr or denoising)
    :param sr_factor: super-resolution factor (optional)
    """
    assert kind in ["swinir", "dip", "pnp", "bm3d", "id", "up"]

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
        img_shape = (3, 48, 48)
        model = DIPModel(physics=physics, sr_factor=sr_factor)
    elif kind == "pnp":
        noise_level_img = noise_level / 255
        early_stop = True
        max_iter = 100

        model = PnPModel(
            physics,
            noise_level_img,
            early_stop=early_stop,
            max_iter=max_iter,
            device=device,
            channels=channels,
        )
    elif kind == "bm3d":
        model = BM3DModel(physics=physics, sigma_psd=noise_level / 255)
    elif kind == "id":
        model = Identity()
    elif kind == "up":
        model = Upsample(factor=sr_factor)

    return model

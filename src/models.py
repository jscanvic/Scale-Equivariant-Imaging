from deepinv.models import SwinIR
from torch import nn
from torch.nn.parallel import DataParallel


def get_model(task, sr_factor=None, device=None, data_parallel_devices=None):
    """
    Get a model with randomly initialized weights for the given task

    :param task: task to perform (i.e. sr or denoising)
    :param sr_factor: super-resolution factor (optional)
    """
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
    if data_parallel_devices is not None:
        devices = data_parallel_devices
        model = DataParallel(model, device_ids=devices, output_device=device)
    return model

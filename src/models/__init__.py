from deepinv.models import SwinIR
from torch import nn
from torch.nn.parallel import DataParallel
from torch.nn import Module

# NOTE: The file structure should be way simpler.
from .convolutional import ConvolutionalModel
from .pnp import PnPModel
from .dip import DeepImagePrior
from .bm3d_deblurring import BM3D
from .upsample import Upsample
from .diffpir import DiffPIR
from .dps import DPS
from .tv import TV


class Identity(Module):
    def forward(self, y):
        return y


class InverseFilter(Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, y):
        return self.physics.A_dagger(y)


class ProposedModel(Module):
    def __init__(
        self,
        blueprint,
        architecture,
        sampling_rate,
    ):
        super().__init__()
        if architecture == "Transformer":
            upsampler = "pixelshuffle" if sampling_rate > 1 else None
            self.model = SwinIR(
                upscale=sampling_rate,
                upsampler=upsampler,
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
                img_range=1.0,
                resi_connection="1conv",
                pretrained=None,
            )
        elif architecture == "Convolutional":
            self.model = ConvolutionalModel(
                in_channels=3,
                upsampling_rate=sampling_rate,
                **blueprint[ConvolutionalModel.__name__],
            )
        else:
            raise ValueError(f"Unknown model kind: {architecture}")

    def forward(self, y):
        return self.model(y)

    def get_backbone(self):
        return self.model


class Model(Module):
    def __init__(
        self,
        blueprint, kind, physics,
        task,
        sr_factor,
        device,
        noise_level,
        data_parallel_devices,
    ):
        super().__init__()
        sampling_rate = sr_factor if task == "sr" else 1
        if kind == "Proposed":
            self.model = ProposedModel(
                blueprint=blueprint,
                sampling_rate=sampling_rate,
                **blueprint[ProposedModel.__name__],
            )
        elif kind == "DeepImagePrior":
            self.model = DeepImagePrior(
                physics=physics,
                sr_factor=sr_factor,
                **blueprint[DeepImagePrior.__name__],
            )
        elif kind == "PlugAndPlay":
            self.model = PnPModel(
                channels=3,
                early_stop=True,
                physics=physics,
                noise_level_img=noise_level / 255,
                device=device,
            )
        elif kind == "BM3D":
            self.model = BM3D(physics=physics, sigma_psd=noise_level / 255)
        elif kind == "DiffPIR":
            self.model = DiffPIR(physics=physics)
        elif kind == "DiffPIR_DiffUNet":
            self.model = DiffPIR(physics=physics, model="DiffUNet")
        elif kind == "DPS":
            self.model = DPS(physics=physics, device=device)
        elif kind == "TV":
            self.model = TV(physics=physics, **blueprint[TV.__name__])
        elif kind == "Identity":
            self.model = Identity()
        elif kind == "InverseFilter":
            self.model = InverseFilter(physics=physics)
        elif kind == "Upsample":
            self.model = Upsample(factor=sr_factor)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        if data_parallel_devices is not None:
            self.model = DataParallel(
                self.model, device_ids=data_parallel_devices, output_device=device
            )

    # NOTE: It'd be better to avoid using args.
    def forward(self, x, *args):
        return self.model(x)

    def get_backbone(self):
        model = self.model

        if isinstance(model, DataParallel):
            model = model.module

        if isinstance(model, ProposedModel):
            backbone = model.get_backbone()
        else:
            backbone = model

        return backbone

    def get_weights(self):
        backbone = self.get_backbone()
        return backbone.state_dict()

    def load_weights(self, state_dict):
        backbone = self.get_backbone()
        backbone.load_state_dict(state_dict)


def get_model(
    args,
    physics,
    device,
):
    data_parallel_devices = (
        args.data_parallel_devices.split(",")
        if args.data_parallel_devices is not None
        else None
    )

    blueprint = {}
    blueprint[ConvolutionalModel.__name__] = {
        "residual": args.ConvolutionalModel__residual,
        "inner_residual": args.ConvolutionalModel__inner_residual,
        "num_conv_blocks": args.ConvolutionalModel__num_conv_blocks,
        "inout_convs": args.ConvolutionalModel__inout_convs,
        "hidden_channels": args.ConvolutionalModel__hidden_channels,
        "scales": args.ConvolutionalModel__scales,
    }

    if args.model_kind == "DeepImagePrior":
        if hasattr(args, "dip_iterations") and args.dip_iterations is not None:
            dip_iterations = args.dip_iterations
        else:
            if args.task == "deblurring" and "Gaussian" in args.kernel:
                dip_iterations = 4000
            elif args.task == "deblurring":
                dip_iterations = 1000
            elif args.task == "sr":
                dip_iterations = 1000
    else:
        dip_iterations = None
    blueprint[DeepImagePrior.__name__] = {
        "iterations": dip_iterations,
    }

    blueprint[TV.__name__] = {
        "lambd": getattr(args, "tv_lambd", None),
        "max_iter": getattr(args, "tv_max_iter", None),
    }

    blueprint[Model.__name__] = {
        "task": args.task,
        "sr_factor": args.sr_factor,
        "noise_level": args.noise_level,
        "kind": args.model_kind,
    }

    blueprint[ProposedModel.__name__] = {
        "architecture": args.ProposedModel__architecture,
    }

    model = Model(
        blueprint=blueprint,
        physics=physics,
        device=device,
        data_parallel_devices=data_parallel_devices,
        **blueprint[Model.__name__],
    )

    return model

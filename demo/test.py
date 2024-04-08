# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

from datasets import TestDataset
from metrics import psnr_fn, ssim_fn
from models import get_model
from physics import get_physics

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="div2k")
parser.add_argument("--task", type=str)
parser.add_argument("--sr_filter", type=str, default="bicubic")
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--weights", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--download", action="store_true")
parser.add_argument("--model_kind", type=str, default="swinir")
parser.add_argument("--Ax_metrics", action="store_true")
parser.add_argument("--dataset_max_size", type=int, default=None)
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--all_images", action="store_true")
parser.add_argument("--dataset_offset", type=int, default=None)
parser.add_argument("--indices", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--save_psf", action="store_true")
parser.add_argument("--dip_iterations", type=int, default=None)
parser.add_argument("--noise2inverse", action="store_true")
parser.add_argument("--print_all_metrics", action="store_true")
parser.add_argument("--resize_gt", type=int, default=None)
parser.add_argument("--no_resize", action="store_true")
parser.add_argument("--r2r", action="store_true")
parser.add_argument("--r2r_itercount", type=int, default=1)
args = parser.parse_args()

physics = get_physics(
    args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
    sr_filter=args.sr_filter,
)

model_kind = args.model_kind
channels = 3
if args.dip_iterations is not None:
    dip_iterations = args.dip_iterations
else:
    if args.task == "deblurring" and "Gaussian" in args.kernel:
        dip_iterations = 4000
    elif args.task == "deblurring":
        dip_iterations = 1000
    elif args.task == "sr":
        dip_iterations = 1000

model = get_model(
    args.task,
    args.sr_factor,
    noise_level=args.noise_level,
    physics=physics,
    device=args.device,
    kind=model_kind,
    channels=channels,
    dip_iterations=dip_iterations,
)
model.to(args.device)
model.eval()

if args.weights is not None:
    if os.path.exists(args.weights):
        weights = torch.load(args.weights, map_location=args.device)
        if "params" in weights:
            weights = weights["params"]
    else:
        weights_url = f"https://huggingface.co/jscanvic/scale-equivariant-imaging/resolve/main/{args.weights}.pt?download=true"
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location=args.device
        )

    model.load_state_dict(weights)

resize = 256
if args.no_resize:
    resize = None
elif args.resize_gt is not None:
    resize = args.resize_gt
force_rgb = False
if args.dataset == "ct":
    force_rgb = True

method = None
if args.noise2inverse:
    method = "noise2inverse"

dataset = TestDataset(
    root="./datasets",
    split=args.split,
    physics=physics,
    resize=resize,
    device=args.device,
    download=args.download,
    dataset=args.dataset,
    max_size=args.dataset_max_size,
    force_rgb=force_rgb,
    offset=args.dataset_offset,
    method=method
)

psnr_list = []
ssim_list = []

psnr_Ax_list = []
ssim_Ax_list = []

if args.save_psf:
    from deepinv.physics import Blur
    from torchvision.utils import save_image

    assert args.out_dir is not None
    assert isinstance(physics, Blur)

    kernel = physics.filter
    assert kernel.dim() == 4
    kernel = kernel.squeeze(0).squeeze(0)
    kernel = kernel / kernel.max()

    os.makedirs(args.out_dir, exist_ok=True)
    save_image(kernel, os.path.join(args.out_dir, "psf.png"))

# testing loop
if args.indices is None:
    indices = range(len(dataset))
else:
    indices = (int(i) for i in args.indices.split(","))

for i in tqdm(indices):
    x, y = dataset[i]
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    if args.dataset == "ct":
        assert x.shape[1] == 3
        assert y.shape[1] == 3

    if args.model_kind != "dip":
        with torch.no_grad():
            if args.noise2inverse:
                from noise2inverse import Noise2InverseModel
                model = Noise2InverseModel(model, physics)
                x_hat = model(y)
                model = model.backbone
            elif args.r2r:
                N = args.r2r_itercount
                x_hat = torch.zeros_like(x)
                for _ in range(N):
                    alpha = .5
                    pert = torch.randn_like(y) * physics.noise_model.sigma
                    x_hat += model(y + alpha * pert)
                x_hat /= N
            else:
                x_hat = model(y)
    else:
        x_hat = model(y).detach()

    assert x_hat.shape[1] in [1, 3]
    y_channel = True if x_hat.shape[1] == 3 else False

    psnr_val = psnr_fn(x_hat, x, y_channel=y_channel).item()
    ssim_val = ssim_fn(x_hat, x, y_channel=y_channel).item()

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    if args.print_all_metrics:
        print(f"METRICS_{i}: PSNR: {psnr_val:.1f}, SSIM: {ssim_val:.3f}")

    if args.Ax_metrics:
        Ax = physics.A(x)
        psnr_Ax_val = psnr_fn(Ax, x, y_channel=y_channel).item()
        ssim_Ax_val = ssim_fn(Ax, x, y_channel=y_channel).item()
        psnr_Ax_list.append(psnr_Ax_val)
        ssim_Ax_list.append(ssim_Ax_val)

    if args.save_images and (args.all_images or i < 5):
        from torchvision.utils import save_image

        assert args.out_dir is not None
        os.makedirs(args.out_dir, exist_ok=True)

        save_image(x, os.path.join(args.out_dir, f"{i}_x.png"))
        save_image(x_hat, os.path.join(args.out_dir, f"{i}_x_hat.png"))
        save_image(y, os.path.join(args.out_dir, f"{i}_y.png"))

N = len(psnr_list)
print(f"N: {N}")

psnr_average = np.mean(psnr_list)
ssim_average = np.mean(ssim_list)
psnr_std = np.std(psnr_list)
ssim_std = np.std(ssim_list)

print(f"PSNR: {psnr_average:.1f}")
print(f"PSNR std: {psnr_std:.1f}")
print(f"SSIM: {ssim_average:.3f}")
print(f"SSIM std: {ssim_std:.3f}")

if args.Ax_metrics:
    psnr_average_Ax = np.mean(psnr_Ax_list)
    ssim_average_Ax = np.mean(ssim_Ax_list)
    psnr_std_Ax = np.std(psnr_Ax_list)
    ssim_std_Ax = np.std(ssim_Ax_list)

    print()
    print(f"PSNR Ax: {psnr_average_Ax:.1f}")
    print(f"PSNR Ax std: {psnr_std_Ax:.1f}")
    print(f"SSIM Ax: {ssim_average_Ax:.3f}")
    print(f"SSIM Ax std: {ssim_std_Ax:.3f}")

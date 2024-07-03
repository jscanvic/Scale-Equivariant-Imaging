# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

import argparse
import os

import torch
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from metrics import psnr_fn, ssim_fn
from models import get_model
from physics import get_physics
from settings import DefaultArgParser

torch.manual_seed(0)
np.random.seed(0)

parser = DefaultArgParser()
parser.add_argument("--weights", type=str)
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--indices", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--save_psf", action="store_true")
parser.add_argument("--dip_iterations", type=int, default=None)
parser.add_argument("--noise2inverse", action="store_true")
parser.add_argument("--print_all_metrics", action="store_true")
parser.add_argument("--r2r", action="store_true")
parser.add_argument("--r2r_itercount", type=int, default=1)
parser.add_argument("--tv_lambd", type=float, default=None)
parser.add_argument("--tv_max_iter", type=int, default=300)
args = parser.parse_args()

physics = get_physics(
    args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
)

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
    args=args,
    physics=physics,
    device=args.device,
    kind=args.model_kind,
    data_parallel_devices=None,
    dip_iterations=dip_iterations,
    tv_lambd=args.tv_lambd,
    tv_max_iter=args.tv_max_iter,
)
model.to(args.device)
model.eval()

if args.weights is not None:
    if os.path.exists(args.weights):
        weights = torch.load(args.weights, map_location=args.device)
    else:
        weights_url = f"https://huggingface.co/jscanvic/scale-equivariant-imaging/resolve/main/{args.weights}.pt?download=true"
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location=args.device
        )

    if "params" in weights:
        weights = weights["params"]

    model.load_state_dict(weights)

dataset = get_dataset(args=args,
                      purpose="test",
                      physics=physics,
                      device=args.device)

psnr_list = []
ssim_list = []

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
    num_indices = max(len(dataset), 100)
    indices = range(num_indices)
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
                    alpha = 0.5
                    pert = torch.randn_like(y) * physics.noise_model.sigma
                    x_hat += model(y + alpha * pert)
                x_hat /= N
            else:
                x_hat = model(y)
    else:
        x_hat = model(y).detach()

    assert x_hat.shape[1] in [1, 3]

    psnr_val = psnr_fn(x_hat, x).item()
    ssim_val = ssim_fn(x_hat, x).item()

    from math import isnan

    if isnan(psnr_val) or isnan(ssim_val):
        breakpoint()

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    if args.print_all_metrics:
        print(f"METRICS_{i}: PSNR: {psnr_val:.1f}, SSIM: {ssim_val:.3f}")

    if args.save_images:
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

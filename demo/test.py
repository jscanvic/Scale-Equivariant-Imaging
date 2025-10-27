# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

import os
from os.path import isdir, dirname
from argparse import BooleanOptionalAction

import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

from datasets import get_dataset
from metrics import compute_metrics
from models import get_model
from physics import get_physics
from settings import DefaultArgParser
from noise2inverse import Noise2InverseModel
from training import get_weights

torch.manual_seed(0)
np.random.seed(0)

parser = DefaultArgParser()
parser.add_argument("--weights", type=str)
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
parser.add_argument("--GroundTruthDataset__split", type=str, default="val")
parser.add_argument(
    "--SyntheticDataset__deterministic_measurements",
    action=BooleanOptionalAction,
    default=True,
)
parser.add_argument("--memoize_gt", action=BooleanOptionalAction, default=False)
args = parser.parse_args()

if not isdir(args.dataset):
    physics = get_physics(args, device=args.device)
else:
    physics = None

model = get_model(
    args=args,
    physics=physics,
    device=args.device,
)
model.to(args.device)
model.eval()

if args.weights is not None:
    weights = get_weights(args.weights, args.device)
    model.load_weights(weights)

basename_table = {}
if isdir(args.dataset):
    from glob import glob
    from os.path import basename
    from torchvision.io import read_image
    dataset = []
    for i, f in enumerate(glob(os.path.join(args.dataset, "*.png"))):
        y = read_image(f)
        y = y.to(args.device)
        y = y.float() / 255.0
        # Discard the alpha channel if it exists
        y = y[:3, :, :]
        x = None
        dataset.append((x, y))
        basename_table[i] = basename(f)
else:
    dataset = get_dataset(args=args, purpose="test", physics=physics, device=args.device, _HOTFIX=False)

psnr_list = []
ssim_list = []
lpips_list = []

if args.save_psf:
    from torchvision.utils import save_image

    assert args.out_dir is not None
    assert physics.task == "deblurring"

    kernel = physics.filter
    assert kernel.dim() == 4
    kernel = kernel.squeeze(0).squeeze(0)
    kernel = kernel / kernel.max()

    os.makedirs(args.out_dir, exist_ok=True)
    save_image(kernel, os.path.join(args.out_dir, "psf.png"))

# testing loop
if args.indices is None:
    num_indices = len(dataset)
    indices = range(num_indices)
else:
    indices = (int(i) for i in args.indices.split(","))

for i in tqdm(indices):
    x, y = dataset[i]

    if x is not None:
        x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    if args.model_kind != "dip":
        with torch.no_grad():
            if args.noise2inverse:
                physics_filter = getattr(physics, "filter", None)
                model = Noise2InverseModel(
                    backbone=model,
                    task=physics.task,
                    physics_filter=physics_filter,
                    degradation_inverse_fn=physics.A_dagger,
                )
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

    # Quantize and clamp
    def quantize_and_clamp(im):
        # quantize
        im = (im * 255.0).round() / 255.0
        # clamp
        return im.clamp(0.0, 1.0)

    x = quantize_and_clamp(x) if x is not None else None
    y = quantize_and_clamp(y)
    x_hat = quantize_and_clamp(x_hat)

    # Compute the metrics
    if x is not None:
        psnr_val, ssim_val, lpips_val = compute_metrics(x.squeeze(0), x_hat.squeeze(0))

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

        if args.print_all_metrics:
            print(f"METRICS_{i}: PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LIPS: {lpips_val:.4f}")

    if args.save_images:
        assert args.out_dir is not None

        entry_basename = basename_table.get(i, f"{i}.png")
        if x is not None:
            path = os.path.join(args.out_dir, "ground_truth", entry_basename)
            os.makedirs(dirname(path), exist_ok=True)
            save_image(x, path)

        path = os.path.join(args.out_dir, "predictors", entry_basename)
        os.makedirs(dirname(path), exist_ok=True)
        save_image(y, path)

        path = os.path.join(args.out_dir, "estimates", entry_basename)
        os.makedirs(dirname(path), exist_ok=True)
        save_image(x_hat, path)

N = len(psnr_list)
if N != 0:
    print(f"N: {N}")

    psnr_average = np.mean(psnr_list)
    ssim_average = np.mean(ssim_list)
    lpips_average = np.mean(lpips_list)
    psnr_std = np.std(psnr_list)
    ssim_std = np.std(ssim_list)
    lpips_std = np.std(lpips_list)

    print(f"PSNR: {psnr_average:.2f}")
    print(f"PSNR std: {psnr_std:.2f}")
    print(f"SSIM: {ssim_average:.4f}")
    print(f"SSIM std: {ssim_std:.4f}")
    print(f"LPIPS: {lpips_average:.4f}")
    print(f"LPIPS std: {lpips_std:.4f}")

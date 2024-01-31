# the library needs to be loaded early on to avoid a bug
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
parser.add_argument("--filter_outliers", action="store_true")
parser.add_argument("--shift", action="store_true")
parser.add_argument("--dataset_offset", type=int, default=None)
parser.add_argument("--indices", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

physics = get_physics(
    args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
)

model_kind = args.model_kind
channels = 3
model = get_model(
    args.task,
    args.sr_factor,
    noise_level=args.noise_level,
    physics=physics,
    device=args.device,
    kind=model_kind,
    channels=channels,
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

resize = None if args.task == "sr" else 256
force_rgb = args.dataset == "ct"
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
)

psnr_sum = 0
ssim_sum = 0

psnr_list = []
ssim_list = []

psnr_sum_Ax = 0
ssim_sum_Ax = 0

psnr_Ax_list = []
ssim_Ax_list = []

if args.indices is not None:
    indices = [int(i) for i in args.indices.split(",")]

# testing loop
for i in tqdm(range(len(dataset))):
    if args.indices is not None:
        if i not in indices:
            continue

    x, y = dataset[i]
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    if args.shift:
        x = torch.roll(x, shifts=1, dims=2)
        x = torch.roll(x, shifts=1, dims=3)
        y = torch.roll(y, shifts=1, dims=2)
        y = torch.roll(y, shifts=1, dims=3)

    if args.dataset == "ct":
        if args.filter_outliers:
            avg_intensity = x.mean()
            outlier_mu = .4424
            outlier_sigma = 0.0312
            if abs(avg_intensity - outlier_mu) > 3 * outlier_sigma:
                continue

    if args.dataset == "ct":
        assert x.shape[1] == 3
        assert y.shape[1] == 3

    if args.model_kind != "dip":
        with torch.no_grad():
            x_hat = model(y)
    else:
        x_hat = model(y).detach()

    assert x_hat.shape[1] in [1, 3]
    y_channel = True if x_hat.shape[1] == 3 else False

    if args.model_kind == "swinir" and args.dataset == "ct" and x.shape[1] == 1:
        psnr_val = psnr_fn(x_hat, x.repeat(1, 3, 1, 1), y_channel=y_channel).item()
        ssim_val = ssim_fn(x_hat, x.repeat(1, 3, 1, 1), y_channel=y_channel).item()
    else:
        psnr_val = psnr_fn(x_hat, x, y_channel=y_channel).item()
        ssim_val = ssim_fn(x_hat, x, y_channel=y_channel).item()

    psnr_sum += psnr_val
    ssim_sum += ssim_val
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    if args.Ax_metrics:
        Ax = physics.A(x)
        psnr_Ax_val = psnr_fn(Ax, x, y_channel=y_channel).item()
        ssim_Ax_val = ssim_fn(Ax, x, y_channel=y_channel).item()
        psnr_sum_Ax += psnr_Ax_val
        ssim_sum_Ax += ssim_Ax_val
        psnr_Ax_list.append(psnr_Ax_val)
        ssim_Ax_list.append(ssim_Ax_val)

    if args.save_images and (args.all_images or i < 5):
        assert args.out_dir is not None
        os.makedirs(args.out_dir, exist_ok=True)

        import cv2

        x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
        x_hat = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()

        x = np.clip(x, 0, 1)
        x_hat = np.clip(x_hat, 0, 1)
        y = np.clip(y, 0, 1)

        x = (x * 255).astype(np.uint8)
        x_hat = (x_hat * 255).astype(np.uint8)
        y = (y * 255).astype(np.uint8)

        # rgb -> bgr
        x = x[..., ::-1]
        x_hat = x_hat[..., ::-1]
        y = y[..., ::-1]

        cv2.imwrite(os.path.join(args.out_dir, f"{i}_x.png"), x)
        cv2.imwrite(os.path.join(args.out_dir, f"{i}_x_hat.png"), x_hat)
        cv2.imwrite(os.path.join(args.out_dir, f"{i}_y.png"), y)


N = len(psnr_list)
print(f"N: {N}")

psnr_average = psnr_sum / N
ssim_average = ssim_sum / N
psnr_std = np.std(psnr_list)
ssim_std = np.std(ssim_list)

print(f"PSNR: {psnr_average:.1f}")
print(f"PSNR std: {psnr_std:.1f}")
print(f"SSIM: {ssim_average:.3f}")
print(f"SSIM std: {ssim_std:.3f}")

if args.Ax_metrics:
    psnr_average_Ax = psnr_sum_Ax / N
    ssim_average_Ax = ssim_sum_Ax / N
    psnr_std_Ax = np.std(psnr_Ax_list)
    ssim_std_Ax = np.std(ssim_Ax_list)

    print()
    print(f"PSNR Ax: {psnr_average_Ax:.1f}")
    print(f"PSNR Ax std: {psnr_std_Ax:.1f}")
    print(f"SSIM Ax: {ssim_average_Ax:.3f}")
    print(f"SSIM Ax std: {ssim_std_Ax:.3f}")

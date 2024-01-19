import argparse
import os

import torch
from tqdm import tqdm
import cv2
import numpy as np

from datasets import TestDataset
from metrics import psnr_fn, ssim_fn
from models import get_model
from physics import get_physics

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str)
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--weights", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--download", action="store_true")
parser.add_argument("--save_images", action="store_true")
parser.add_argument("--all_images", action="store_true")
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

model = get_model(args.task, args.sr_factor)
model.to(args.device)
model.eval()

if args.weights is not None:
    if os.path.exists(args.weights):
        weights = torch.load(args.weights, map_location=args.device)
        if "params" in weights:
            weights = weights["params"]
        # if the weights are from a DataParallel model, remove the "model." prefix
        new_weights = {}
        prefix = "module."
        for k, v in weights.items():
            if k.startswith(prefix):
                new_weights[k[len(prefix):]] = v
        weights = new_weights
    else:
        weights_url = f"https://huggingface.co/jscanvic/scale-equivariant-imaging/resolve/main/{args.weights}.pt?download=true"
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location=args.device
        )

    model.load_state_dict(weights)

physics = get_physics(
    args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
)

resize = None if args.task == "sr" else 256
dataset = TestDataset(
    root="./datasets",
    split=args.split,
    physics=physics,
    resize=resize,
    device=args.device,
    download=args.download,
)

psnr_sum = 0
ssim_sum = 0

# testing loop
for i in tqdm(range(len(dataset))):
    x, y = dataset[i]
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    with torch.no_grad():
        x_hat = model(y)

    psnr_sum += psnr_fn(x_hat, x, y_channel=True)
    ssim_sum += ssim_fn(x_hat, x, y_channel=True)

    if args.save_images and (args.all_images or i < 5):
        assert args.out_dir is not None
        os.makedirs(args.out_dir, exist_ok=True)

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

psnr_average = psnr_sum / len(dataset)
ssim_average = ssim_sum / len(dataset)

print(f"PSNR: {psnr_average:.1f}")
print(f"SSIM: {ssim_average:.3f}")

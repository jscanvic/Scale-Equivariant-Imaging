import os
import pyiqa
from kornia.color import rgb_to_ycbcr
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchvision.transforms import CenterCrop

def psnr_fn(x_hat, x):
    x_hat = rgb_to_ycbcr(x_hat)[0:1, :, :]
    x = rgb_to_ycbcr(x)[0:1, :, :]
    return peak_signal_noise_ratio(x_hat, x, data_range=1.0)

def ssim_fn(x_hat, x):
    x_hat = rgb_to_ycbcr(x_hat)[None, 0:1, :, :]
    x = rgb_to_ycbcr(x)[None, 0:1, :, :]
    return structural_similarity_index_measure(x_hat, x, data_range=1.0)

def lpips_fn(x_hat, x):
    if x.ndim != x_hat.ndim:
        raise ValueError(f"Dimension mismatch: x.ndim={x.ndim}, x_hat.ndim={x_hat.ndim}")
    if x.ndim != 4:
        x = x.unsqueeze(0)
    if x_hat.ndim != 4:
        x_hat = x_hat.unsqueeze(0)
    if "_lpips_fn" not in globals():
        global _lpips_fn
        _lpips_fn = pyiqa.create_metric("lpips")
    return _lpips_fn(x_hat, x)

def register_fn(x, x_hat):
    if x.shape[-2] != x_hat.shape[-2] or x.shape[-1] != x_hat.shape[-1]:
        hmin = min(x.shape[-2], x_hat.shape[-2])
        wmin = min(x.shape[-1], x_hat.shape[-1])
        f_crop = CenterCrop((hmin, wmin))
        x = f_crop(x)
        x_hat = f_crop(x_hat)
    return x, x_hat

def compute_metrics(x, x_hat):
    x, x_hat = register_fn(x, x_hat)
    psnr = psnr_fn(x, x_hat).item()
    ssim = ssim_fn(x, x_hat).item()
    lpips = lpips_fn(x, x_hat).item()
    return psnr, ssim, lpips

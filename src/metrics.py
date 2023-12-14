from kornia.color import rgb_to_ycbcr
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def psnr_fn(x_hat, x, y_channel=False):
    """
    Compute the PSNR between two images

    :param torch.Tensor x_hat: reconstructed image
    :param torch.Tensor x: ground truth image
    :param bool y_channel: compute PSNR on the Y channel in CbCr space
    """
    if y_channel:
        x_hat = rgb_to_ycbcr(x_hat)[:, 0:1, :, :]
        x = rgb_to_ycbcr(x)[:, 0:1, :, :]
    return peak_signal_noise_ratio(x_hat, x, data_range=1.0)


def ssim_fn(x_hat, x, y_channel=False):
    """
    Compute the SSIM between two images

    :param torch.Tensor x_hat: reconstructed image
    :param torch.Tensor x: ground truth image
    :param bool y_channel: compute SSIM on the Y channel in CbCr space
    """
    if y_channel:
        x_hat = rgb_to_ycbcr(x_hat)[:, 0:1, :, :]
        x = rgb_to_ycbcr(x)[:, 0:1, :, :]
    return structural_similarity_index_measure(x_hat, x, data_range=1.0)

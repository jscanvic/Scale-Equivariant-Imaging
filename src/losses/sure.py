# Obtained from https://github.com/deepinv/deepinv/blob/9c8366497940916e70f1abb9e63c7310d85af9f7/deepinv/loss/sure.py
import torch
import torch.nn as nn
import numpy as np


def mc_div(y1, y, f, physics, tau, crop_size=None):
    r"""
    Monte-Carlo estimation for the divergence of A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :return: (float) hutch divergence.
    """
    if crop_size is None:
        b = torch.randn_like(y)
    else:
        # shape of the inner part of the measurements
        ip_shape = (y.size(0), y.size(1), y.size(2) - crop_size, y.size(3) - crop_size)
        # half crop size
        hcs = crop_size // 2
        b = torch.zeros_like(y)
        b[:, :, hcs:-hcs, hcs:-hcs] = torch.randn(*ip_shape, device=y.device, dtype=y.dtype)

    y2 = physics.A(f(y + b * tau, physics))

    out = b * (y2 - y1) / tau

    if crop_size is not None:
        # half crop size
        hcs = crop_size // 2
        out = out[:, :, hcs:-hcs, hcs:-hcs]

    out = out.mean()
    return out


class SureGaussianLoss(nn.Module):
    r"""
    SURE loss for Gaussian noise

    The loss is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|y - A\inverse{y}\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(A\inverse{y+\tau b_i} -
        A\inverse{y}\right)

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`A` is the forward operator,
    :math:`b\sim\mathcal{N}(0,I)` and :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence.

    This loss approximates the divergence of :math:`A\inverse{y}` (in the original SURE loss)
    using the Monte Carlo approximation in
    https://ieeexplore.ieee.org/abstract/document/4099398/

    If the measurement data is truly Gaussian with standard deviation :math:`\sigma`,
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-A\inverse{y}\|_2^2`
    where :math:`z` is the noiseless measurement.

    .. warning::

        The loss can be sensitive to the choice of :math:`\tau`, which should be proportional to the size of :math:`y`.
        The default value of 0.01 is adapted to :math:`y` vectors with entries in :math:`[0,1]`.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    """

    def __init__(self, sigma, tau=1e-2, measurements_crop_size=None, cropped_div=False, averaged_cst=False):
        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = sigma**2
        self.tau = tau
        self.measurements_crop_size = measurements_crop_size
        self.cropped_div = cropped_div
        self.averaged_cst = averaged_cst

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE Loss.
        :param torch.Tensor y: Measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module, deepinv.models.Denoiser model: Reconstruction network.
        :return: (float) SURE loss.
        """

        y1 = physics.A(x_net)

        if not self.cropped_div:
            div = mc_div(y1, y, model, physics, self.tau)
        else:
            div = mc_div(y1, y, model, physics, self.tau, crop_size=self.measurements_crop_size)
        div = 2 * self.sigma2 * div

        mse = y1 - y
        if self.measurements_crop_size is not None:
            half_crop_size = self.measurements_crop_size // 2
            mse = mse[:, :, half_crop_size:-half_crop_size, half_crop_size:-half_crop_size]
        mse = mse.pow(2).mean()

        if not self.averaged_cst:
            loss_sure = mse + div - self.sigma2
        else:
            loss_sure = mse + div - self.sigma2 / y.size(0)

        return loss_sure

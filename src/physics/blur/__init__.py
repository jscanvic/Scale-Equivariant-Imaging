# Code obtained from
# https://github.com/deepinv/deepinv/blob/a1ef4a8a8de0eacb1c0d0fb463a721de7827415e/deepinv/physics/blur.py
import torch
from torch.nn.functional import pad
import torch.nn.functional as F
from deepinv.physics.forward import LinearPhysics
from deepinv.physics import adjoint_function

def extend_filter(filter):
    b, c, h, w = filter.shape
    w_new = w
    h_new = h

    offset_w = 0
    offset_h = 0

    if w == 1:
        w_new = 3
        offset_w = 1
    elif w % 2 == 0:
        w_new += 1

    if h == 1:
        h_new = 3
        offset_h = 1
    elif h % 2 == 0:
        h_new += 1

    out = torch.zeros((b, c, h_new, w_new), device=filter.device)
    out[:, :, offset_h : h + offset_h, offset_w : w + offset_w] = filter
    return out


def conv(x, filter, padding):
    r"""
    Convolution of x and filter. The transposed of this operation is conv_transpose(x, filter, padding)

    :param x: (torch.Tensor) Image of size (B,C,W,H).
    :param filter: (torch.Tensor) Filter of size (1,C,W,H) for colour filtering or (1,1,W,H) for filtering each channel with the same filter.
    :param padding: (string) options = 'valid','circular','replicate','reflect'. If padding='valid' the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    """
    b, c, h, w = x.shape

    filter = filter.flip(-1).flip(
        -2
    )  # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1) / 2
    pw = (filter.shape[3] - 1) / 2

    if padding == "valid":
        h_out = int(h - 2 * ph)
        w_out = int(w - 2 * pw)
    else:
        h_out = h
        w_out = w
        pw = int(pw)
        ph = int(ph)
        x = F.pad(x, (pw, pw, ph, ph), mode=padding, value=0)

    if filter.shape[1] == 1:
        y = torch.zeros((b, c, h_out, w_out), device=x.device)
        for i in range(b):
            for j in range(c):
                y[i, j, :, :] = F.conv2d(
                    x[i, j, :, :].unsqueeze(0).unsqueeze(1), filter, padding="valid"
                ).unsqueeze(1)
    else:
        y = F.conv2d(x, filter, padding="valid")

    return y


def conv_transpose(y, filter, padding):
    r"""
    Tranposed convolution of x and filter. The transposed of this operation is conv(x, filter, padding)

    :param torch.tensor x: Image of size (B,C,W,H).
    :param torch.tensor filter: Filter of size (1,C,W,H) for colour filtering or (1,C,W,H) for filtering each channel with the same filter.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    """

    b, c, h, w = y.shape

    filter = filter.flip(-1).flip(
        -2
    )  # In order to perform convolution and not correlation like Pytorch native conv

    filter = extend_filter(filter)

    ph = (filter.shape[2] - 1) / 2
    pw = (filter.shape[3] - 1) / 2

    h_out = int(h + 2 * ph)
    w_out = int(w + 2 * pw)
    pw = int(pw)
    ph = int(ph)

    x = torch.zeros((b, c, h_out, w_out), device=y.device)
    if filter.shape[1] == 1:
        for i in range(b):
            if filter.shape[0] > 1:
                f = filter[i, :, :, :].unsqueeze(0)
            else:
                f = filter

            for j in range(c):
                x[i, j, :, :] = F.conv_transpose2d(
                    y[i, j, :, :].unsqueeze(0).unsqueeze(1), f
                )
    else:
        x = F.conv_transpose2d(y, filter)

    if padding == "valid":
        out = x
    elif padding == "zero":
        out = x[:, :, ph:-ph, pw:-pw]
    elif padding == "circular":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, :ph, :] += x[:, :, -ph:, pw:-pw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw:-pw]
        out[:, :, :, :pw] += x[:, :, ph:-ph, -pw:]
        out[:, :, :, -pw:] += x[:, :, ph:-ph, :pw]
        # corners
        out[:, :, :ph, :pw] += x[:, :, -ph:, -pw:]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, :ph, -pw:] += x[:, :, -ph:, :pw]
        out[:, :, -ph:, :pw] += x[:, :, :ph, -pw:]

    elif padding == "reflect":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw:-pw].flip(dims=(2,))
        out[:, :, -ph - 1 : -1, :] += x[:, :, -ph:, pw:-pw].flip(dims=(2,))
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph:-ph, :pw].flip(dims=(3,))
        out[:, :, :, -pw - 1 : -1] += x[:, :, ph:-ph, -pw:].flip(dims=(3,))
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, -pw - 1 : -1] += x[:, :, -ph:, -pw:].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, 1 : 1 + pw] += x[:, :, -ph:, :pw].flip(dims=(2, 3))
        out[:, :, 1 : 1 + ph, -pw - 1 : -1] += x[:, :, :ph, -pw:].flip(dims=(2, 3))

    elif padding == "replicate":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw:-pw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph:, pw:-pw].sum(2)
        out[:, :, :, 0] += x[:, :, ph:-ph, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph:-ph, -pw:].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph:, -pw:].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph:, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw:].sum(3).sum(2)
    return out


class Blur(LinearPhysics):
    r"""

    Blur operator.

    This forward operator performs

    .. math:: y = w*x

    where :math:`*` denotes convolution and :math:`w` is a filter.

    This class uses :meth:`torch.nn.functional.conv2d` for performing the convolutions.

    :param torch.Tensor filter: Tensor of size (1, 1, H, W) or (1, C, H, W) containing the blur filter, e.g., :meth:`deepinv.physics.blur.gaussian_blur`.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``. If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.
    :param str device: cpu or cuda.

    """

    def __init__(self, filter, padding="circular", device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
        self.device = device
        self.filter = torch.nn.Parameter(filter, requires_grad=False).to(device)

    def A(self, x):
        return conv(x, self.filter, self.padding)

    def A_adjoint(self, y):
        return conv_transpose(y, self.filter, self.padding)


class BlurV2(LinearPhysics):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.fft_norm = "backward"

    def A(self, x):
        y = x

        shape = y.shape[-2:]

        kernel = self.kernel.to(y.device, y.dtype)
        psf = torch.zeros(shape, device=y.device, dtype=y.dtype)
        psf[: kernel.shape[-2], : kernel.shape[-1]] = kernel
        psf = psf.roll(
                (-(kernel.shape[-2] // 2), -(kernel.shape[-1] // 2)),
                dims=(-2, -1)
            )
        otf = torch.fft.rfft2(psf, dim=(-2, -1), norm=self.fft_norm)

        y = torch.fft.rfft2(y, dim=(-2, -1), norm=self.fft_norm)
        y = otf.broadcast_to(y.shape) * y
        y = torch.fft.irfft2(y, dim=(-2, -1), s=shape, norm=self.fft_norm)

        return y

    def A_adjoint(self, y):
        fn = adjoint_function(self.A, y.shape, device=y.device, dtype=y.dtype)
        return fn(y)

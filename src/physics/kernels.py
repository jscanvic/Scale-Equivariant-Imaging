import torch

_table = {
    "Gaussian_R1": ("gaussian", 1),
    "Gaussian_R2": ("gaussian", 2),
    "Gaussian_R3": ("gaussian", 3),
    "Box_R2": ("box", 2),
    "Box_R3": ("box", 3),
    "Box_R4": ("box", 4),
}


def get_kernel(name, dtype=torch.float64):
    assert name in _table, f"Unsupported kernel: {name}"
    blur_type, blur_level = _table[name]
    if blur_type == "gaussian":
        kernel_size = blur_level * 6 + 1
        u = torch.arange(kernel_size, dtype=dtype)
        u = u - (kernel_size - 1) / 2
        v = u
        U, V = torch.meshgrid(u, v, indexing="ij")
        kernel = torch.exp(-(U**2 + V**2) / (2 * blur_level**2))
        kernel = kernel / kernel.sum()
    elif blur_type == "box":
        kernel_size = blur_level * 2 + 1
        kernel = torch.ones(kernel_size, kernel_size, dtype=dtype)
        kernel = kernel / kernel.sum()
    return kernel

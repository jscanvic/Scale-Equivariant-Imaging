from torch.random import fork_rng as torch_fork_rng


def fork_rng(enabled):
    return torch_fork_rng(enabled=enabled)

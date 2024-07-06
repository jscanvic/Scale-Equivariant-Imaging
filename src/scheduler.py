import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR


def get_lr_scheduler(optimizer, epochs, lr_scheduler_kind):
    if lr_scheduler_kind == "multi_step_decay":
        milestones = [
            epochs * 50 // 100,
            epochs * 80 // 100,
            epochs * 90 // 100,
            epochs * 95 // 100,
        ]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    elif lr_scheduler_kind == "delayed_linear_decay":
        schedulers = [
            LinearLR(optimizer, start_factor=1, end_factor=1, total_iters=epochs // 2),
            LinearLR(
                optimizer, start_factor=1, end_factor=1e-2, total_iters=epochs // 2 - 1
            ),
        ]
        scheduler = SequentialLR(optimizer, schedulers, [epochs // 2])
    return scheduler

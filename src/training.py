import os

import torch

from models import get_weights


def save_training_state(epoch, model, optimizer, scheduler, state_path):
    """
    Save the training state to a file

    :param epoch: current epoch
    :param model: model to save
    :param optimizer: optimizer to save
    :param scheduler: scheduler to save
    :param state_path: path where to save the state
    """
    save_dir = os.path.dirname(state_path)
    os.makedirs(save_dir, exist_ok=True)

    weights = get_weights(model)

    print(f"writing the training state to the file {state_path}")
    torch.save(
        {
            "epoch": epoch,
            "params": weights,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        state_path,
    )

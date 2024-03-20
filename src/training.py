import os

import torch


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

    if not isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.state_dict()
    else:
        model_state_dict = model.module.state_dict()

    print(f"writing the training state to the file {state_path}")
    torch.save(
        {
            "epoch": epoch,
            "params": model_state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        state_path,
    )

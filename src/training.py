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

    weights = model.get_weights()

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


def get_weights(weights_name, device):
    if os.path.exists(weights_name):
        weights = torch.load(weights_name, map_location=device)
    else:
        weights_url = f"https://huggingface.co/jscanvic/scale-equivariant-imaging/resolve/main/{weights_name}.pt?download=true"
        weights = torch.hub.load_state_dict_from_url(
            weights_url, map_location=device
        )

    if "params" in weights:
        weights = weights["params"]

    return weights

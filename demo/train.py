import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from deepinv.utils import AverageMeter, ProgressMeter

from datasets import TrainingDataset, EvalDataset
from losses import get_losses
from metrics import psnr_fn
from models import get_model
from training import save_training_state
from physics import get_physics

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str)
parser.add_argument("--stop_gradient", action="store_true")
parser.add_argument("--task", type=str)
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--data_parallel_devices", type=str, default=None)
parser.add_argument("--download", action="store_true")
parser.add_argument("--byol_decay", type=float, default=0.99)
args = parser.parse_args()

assert args.method in [
    "proposed",
    "sup",
    "css",
    "ei-rotate",
    "ei-shift",
], "Unsupported training method"
assert args.task in ["sr", "deblurring"], "Unsupported task"

data_parallel_devices = args.data_parallel_devices.split(",") if args.data_parallel_devices is not None else None

model = get_model(args.task, args.sr_factor, device=args.device, data_parallel_devices=data_parallel_devices)
model.to(args.device)
model.train()

byol_model = get_model(args.task, args.sr_factor, device=args.device, data_parallel_devices=data_parallel_devices)
byol_model.to(args.device)
byol_model.train()
byol_params = dict(byol_model.named_parameters())

physics = get_physics(
    task=args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
)

dataset_root = "./datasets"
css = args.method == "css"
resize = None if args.sr_factor == "sr" else 256
training_dataset = TrainingDataset(
    dataset_root,
    physics,
    resize=resize,
    css=css,
    download=args.download,
    device=args.device,
)
eval_dataset = EvalDataset(
    dataset_root, physics, resize=resize, download=args.download, device=args.device
)

losses = get_losses(
    args.method, args.noise_level, args.stop_gradient, byol_params=byol_params
)

batch_size = 16 if args.task == "sr" else 8
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

lr = 2e-4 if args.task == "sr" else 5e-4
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = MultiStepLR(optimizer, milestones=[250, 400, 450, 475], gamma=0.5)

# training meters used for monitoring the training process
loss_meter = AverageMeter("Training_Loss", ":.2e")
psnr_meter = AverageMeter("Eval_PSNR", ":.2f")

epochs = 500
progress = ProgressMeter(epochs, [loss_meter, psnr_meter])

# training loop
for epoch in range(epochs):
    # set training meters to zero
    loss_meter.reset()
    psnr_meter.reset()

    # stochastic gradient descent step
    for x, y in training_dataloader:
        x, y = x.to(args.device), y.to(args.device)

        optimizer.zero_grad()

        x_hat = model(y)

        training_loss = 0
        for loss_fn in losses:
            training_loss += loss_fn(
                x=x, x_net=x_hat, y=y, physics=physics, model=model
            )

        training_loss.backward()
        optimizer.step()

        # byol update
        params = dict(model.named_parameters())
        for k, byol_param in byol_params.items():
            param = params[k]
            byol_param.data = (
                args.byol_decay * byol_param.data + (1 - args.byol_decay) * param.data
            )

        loss_meter.update(training_loss.item())

    scheduler.step()

    # evaluate the model every 5 epochs
    if epoch % 5 == 0:
        for x, y in eval_dataloader:
            x = x.to(args.device)
            y = y.to(args.device)

            with torch.no_grad():
                x_hat = model(y)

            cur_psnr = psnr_fn(x_hat, x)
            psnr_meter.update(cur_psnr.item())

    # report the training progress
    progress.display(epoch)

    # save the training state every 50 epochs and after training completion
    if (epoch % 50 == 0) or (epoch == epochs - 1):
        checkpoint_path = f"{args.out_dir}/checkpoints/ckp_{epoch}.pt"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
torch.save(model.state_dict(), weights_path)

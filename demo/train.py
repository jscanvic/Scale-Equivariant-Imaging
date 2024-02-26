# the library needs to be loaded early on to avoid a bug
import bm3d

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
parser.add_argument("--dataset", type=str, default="div2k")
parser.add_argument("--method", type=str)
parser.add_argument("--stop_gradient", action="store_true")
parser.add_argument("--task", type=str)
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--download", action="store_true")
args = parser.parse_args()

assert args.method in [
    "proposed",
    "sup",
    "css",
    "ei-rotate",
    "ei-shift",
    "dip",
    "pnp",
    "noise2inverse"
], "Unsupported training method"
assert args.task in ["sr", "deblurring"], "Unsupported task"

physics = get_physics(
    task=args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
)

channels = 3
model = get_model(
    args.task, args.sr_factor, physics=physics, device=args.device, kind="swinir",
    channels=channels
)
model.to(args.device)
model.train()

dataset_root = "./datasets"
css = args.method == "css"
resize = None if args.sr_factor == "sr" else 256
force_rgb = args.dataset == "ct"
training_dataset = TrainingDataset(
    dataset_root,
    physics,
    resize=resize,
    css=css,
    download=args.download,
    device=args.device,
    dataset=args.dataset,
    force_rgb=force_rgb,
    method=args.method
)
eval_dataset = EvalDataset(
    dataset_root,
    physics,
    resize=resize,
    download=args.download,
    device=args.device,
    dataset=args.dataset,
    force_rgb=force_rgb
)

losses = get_losses(args.method, args.noise_level, args.stop_gradient)

batch_size = 16 if args.task == "sr" else 8
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

lr = 2e-4 if args.task == "sr" else 5e-4
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = MultiStepLR(optimizer, milestones=[250, 400, 450, 475], gamma=0.5)

# training meters used for monitoring the training process
loss_meter = AverageMeter("Training_Loss", ":.2e")
psnr_meter = AverageMeter("Eval_PSNR", ":.2f")

epochs = 500 if args.dataset != "ct" else 100
if args.dataset == "urban100":
    epochs = 4000
progress = ProgressMeter(epochs, [loss_meter, psnr_meter])

# training loop
for epoch in range(epochs):
    # set training meters to zero
    loss_meter.reset()
    psnr_meter.reset()

    # stochastic gradient descent step
    for x, y in training_dataloader:
        x, y = x.to(args.device), y.to(args.device)
        if args.dataset == "ct":
            assert x.shape[1] == 3
            assert y.shape[1] == 3

        optimizer.zero_grad()

        x_hat = model(y)

        training_loss = 0
        for loss_fn in losses:
            training_loss += loss_fn(
                x=x, x_net=x_hat, y=y, physics=physics, model=model
            )

        training_loss.backward()
        optimizer.step()

        loss_meter.update(training_loss.item())

    scheduler.step()

    # evaluate the model regularly
    eval_interval = 5
    if args.dataset == "ct" and args.method == "proposed":
        if epoch <= 10:
            eval_interval = 1
        elif epoch <= 50:
            eval_interval = 5
        elif epoch <= 100:
            eval_interval = 10
    if epoch % eval_interval == 0:
        for x, y in eval_dataloader:
            x = x.to(args.device)
            y = y.to(args.device)

            with torch.no_grad():
                x_hat = model(y)

            cur_psnr = psnr_fn(x_hat, x)
            psnr_meter.update(cur_psnr.item())

    # report the training progress
    progress.display(epoch)

    # save the training state regularly and after training completion
    checkpoint_interval = 50
    if args.dataset == "urban100":
        if args.method == "proposed":
            if epoch <= 1000:
                checkpoint_interval = 100
            elif epoch <= 2000:
                checkpoint_interval = 200
            elif epoch <= 3000:
                checkpoint_interval = 400
            else:
                checkpoint_interval = 800
        else:
            checkpoint_interval = 400
    elif args.dataset == "ct":
        if args.method == "proposed":
            if epoch <= 10:
                checkpoint_interval = 1
            elif epoch <= 50:
                checkpoint_interval = 5
            else:
                checkpoint_interval = 10
        else:
            checkpoint_interval = 50
    if (epoch % checkpoint_interval == 0) or (epoch == epochs - 1):
        checkpoint_path = f"{args.out_dir}/checkpoints/ckp_{epoch}.pt"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
torch.save(model.state_dict(), weights_path)

# the library needs to be loaded early on to avoid a bug
import bm3d

import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
from deepinv.utils import AverageMeter, ProgressMeter

from datasets import TrainingDataset, EvalDataset
from metrics import psnr_fn
from models import get_model
from training import save_training_state
from physics import get_physics
from noise2inverse import n2i_pair

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="div2k")
parser.add_argument("--stop_gradient", action="store_true")
parser.add_argument("--task", type=str)
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--download", action="store_true")
args = parser.parse_args()

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
css = False
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
# https://github.com/ahendriksen/noise2inverse/blob/master/03_train.ipynb
for epoch in range(epochs):
    # set training meters to zero
    loss_meter.reset()
    psnr_meter.reset()

    # stochastic gradient descent step
    for _, y in training_dataloader:
        y = y.to(args.device)
        inp, tgt = n2i_pair(y, strategy="X:1", num_splits=4)

        output = model(inp)
        training_loss = mse_loss(output, tgt)
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        loss_meter.update(training_loss.item())

    scheduler.step()

    # evaluate the model regularly
    eval_interval = 5
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
        checkpoint_interval = 400
    elif args.dataset == "ct":
        checkpoint_interval = 50
    if (epoch % checkpoint_interval == 0) or (epoch == epochs - 1):
        checkpoint_path = f"{args.out_dir}/checkpoints/ckp_{epoch}.pt"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
torch.save(model.state_dict(), weights_path)

# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
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
from training import save_training_state, get_model_state_dict
from physics import get_physics
from torch.nn.parallel import DataParallel
from noise2inverse import Noise2InverseModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="div2k")
parser.add_argument("--method", type=str)
parser.add_argument("--scale_transforms_antialias", action="store_true")
parser.add_argument(
    "--stop_gradient", default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument("--task", type=str)
parser.add_argument("--sr_factor", type=int, default=None)
parser.add_argument("--sr_filter", type=str, default="bicubic_torch")
parser.add_argument("--kernel", type=str, default=None)
parser.add_argument("--noise_level", type=int)
parser.add_argument("--resize_gt", type=int, default=256)
parser.add_argument("--no_resize_gt", action="store_true")
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--data_parallel_devices", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--download", action="store_true")
parser.add_argument("--sure_alternative", type=str, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--memoize_gt", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--loss_alpha_tradeoff", type=float, default=1.0)
args = parser.parse_args()

data_parallel_devices = (
    args.data_parallel_devices.split(",")
    if args.data_parallel_devices is not None
    else None
)

assert args.method in [
    "proposed",
    "sup",
    "css",
    "ei-rotate",
    "ei-shift",
    "noise2inverse",
], "Unsupported training method"
assert args.task in ["sr", "deblurring"], "Unsupported task"

physics = get_physics(
    task=args.task,
    noise_level=args.noise_level,
    kernel_path=args.kernel,
    sr_factor=args.sr_factor,
    device=args.device,
    sr_filter=args.sr_filter,
)

model = get_model(
    args.task,
    args.sr_factor,
    physics=physics,
    device=args.device,
    kind="swinir",
    data_parallel_devices=data_parallel_devices,
)
model.to(args.device)
model.train()

dataset_root = "./datasets"
css = args.method == "css"
resize_gt = None if args.no_resize_gt else args.resize_gt
force_rgb = True if args.dataset == "ct" else False

training_dataset = TrainingDataset(
    dataset_root,
    physics,
    resize=resize_gt,
    css=css,
    download=args.download,
    device=args.device,
    dataset=args.dataset,
    force_rgb=force_rgb,
    method=args.method,
    memoize_gt=args.memoize_gt,
)
eval_dataset = EvalDataset(
    dataset_root,
    physics,
    resize=resize_gt,
    download=args.download,
    device=args.device,
    dataset=args.dataset,
    force_rgb=force_rgb,
    method=args.method,
    memoize_gt=args.memoize_gt,
)

losses = get_losses(
    args.method,
    args.noise_level,
    args.stop_gradient,
    sure_alternative=args.sure_alternative,
    scale_antialias=args.scale_transforms_antialias,
    alpha_tradeoff=args.loss_alpha_tradeoff,
)

batch_size = args.batch_size or 8
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

lr = 2e-4 if args.task == "sr" else 5e-4
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = MultiStepLR(optimizer, milestones=[250, 400, 450, 475], gamma=0.5)

# training meters used for monitoring the training process
loss_meter = AverageMeter("Training_Loss", ":.2e")
psnr_meter = AverageMeter("Eval_PSNR", ":.2f")

if args.epochs is None:
    epochs = 500 if args.dataset != "ct" else 100
    if args.dataset == "urban100":
        epochs = 4000
else:
    epochs = args.epochs
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

        if args.method == "proposed" and args.sure_alternative == "r2r":
            x_hat = None
        else:
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
                if args.method != "noise2inverse":
                    x_hat = model(y)
                else:
                    model = Noise2InverseModel(model, physics)
                    x_hat = model(y)
                    model = model.backbone

            cur_psnr = psnr_fn(x_hat, x)
            psnr_meter.update(cur_psnr.item())

    # report the training progress
    progress.display(epoch)

    # save the training state regularly and after training completion
    if args.checkpoint_interval is not None:
        checkpoint_interval = args.checkpoint_interval
    else:
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
model_state_dict = get_model_state_dict(model)
torch.save(model_state_dict, weights_path)

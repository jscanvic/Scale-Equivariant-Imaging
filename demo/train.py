# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

import argparse
from argparse import BooleanOptionalAction

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from datetime import datetime
from torchmetrics import MeanMetric

from datasets import TrainingDataset
from losses import get_loss
from metrics import psnr_fn
from models import get_model
from training import save_training_state, get_model_state_dict
from physics import get_physics
from torch.nn.parallel import DataParallel
from noise2inverse import Noise2InverseModel
from settings import DefaultArgParser

parser = DefaultArgParser()
parser.add_argument("--method", type=str)
parser.add_argument("--stop_gradient", action=BooleanOptionalAction, default=True)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--data_parallel_devices", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--sure_alternative", type=str, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--memoize_gt", action=BooleanOptionalAction, default=True)
parser.add_argument("--loss_alpha_tradeoff", type=float, default=1.0)
parser.add_argument("--scale_transforms_antialias", action=BooleanOptionalAction, default=False)
parser.add_argument("--partial_sure", action=BooleanOptionalAction, default=True)
parser.add_argument("--sure_cropped_div", action=BooleanOptionalAction, default=True)
parser.add_argument("--sure_averaged_cst", action=BooleanOptionalAction, default=None)
parser.add_argument("--partial_sure_sr", action=BooleanOptionalAction, default=False)
parser.add_argument("--sure_margin", type=int, default=None)
args = parser.parse_args()

data_parallel_devices = (
    args.data_parallel_devices.split(",")
    if args.data_parallel_devices is not None
    else None
)

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
gt_size = args.gt_size if args.resize_gt else None

training_dataset = TrainingDataset(
    dataset_root,
    physics,
    resize=gt_size,
    css=css,
    download=args.download,
    device=args.device,
    dataset=args.dataset,
    method=args.method,
    memoize_gt=args.memoize_gt,
)

if args.partial_sure:
    if args.sure_margin is not None:
        sure_margin = args.sure_margin
    elif args.task == "deblurring":
        from physics import Blur
        assert isinstance(physics, Blur)

        kernel = physics.filter
        kernel_size = max(kernel.shape[-2], kernel.shape[-1])

        sure_margin = (kernel_size - 1) // 2
    elif args.task == "sr":
        if args.partial_sure_sr:
            assert args.sr_filter == "bicubic_torch"
            sure_margin = 2
        else:
            sure_margin = 0
else:
    assert args.sure_margin is None
    sure_margin = 0

loss = get_loss(args=args, sure_margin=sure_margin)

batch_size = args.batch_size or 8
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

lr = 2e-4 if args.task == "sr" else 5e-4
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = MultiStepLR(optimizer, milestones=[250, 400, 450, 475], gamma=0.5)

training_loss_metric = MeanMetric()

if args.epochs is None:
    epochs = 500 if args.dataset != "ct" else 100
    if args.dataset == "urban100":
        epochs = 4000
else:
    epochs = args.epochs

# training loop
for epoch in range(epochs):
    training_loss_metric.reset()

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

        training_loss = loss(x=x, x_net=x_hat, y=y, physics=physics, model=model)

        training_loss.backward()
        optimizer.step()

        training_loss_metric.update(training_loss.item())

    scheduler.step()

    # log progress to stdout
    epochs_ndigits = len(str(int(epochs)))
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_training_loss = training_loss_metric.compute().item()
    print(f"\t{current_timestamp}\t[{epoch + 1:{epochs_ndigits}d}/{epochs}]\tTraining_Loss: {epoch_training_loss:.2e}")

    # save the training state regularly and after training completion
    if args.checkpoint_interval is not None:
        checkpoint_interval = args.checkpoint_interval
    else:
        checkpoint_interval = 50
        if args.dataset == "urban100":
            checkpoint_interval = 400
        elif args.dataset == "ct":
            checkpoint_interval = 50
    if (epoch % checkpoint_interval == 0) or (epoch == epochs - 1):
        checkpoint_path = f"{args.out_dir}/checkpoints/ckp_{epoch:03}.pt"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
model_state_dict = get_model_state_dict(model)
torch.save(model_state_dict, weights_path)

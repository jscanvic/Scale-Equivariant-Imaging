# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

from argparse import BooleanOptionalAction
import csv
import os
from os.path import isdir

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from datetime import datetime
from torchmetrics import MeanMetric

from datasets import get_dataset
from losses import get_loss
from metrics import psnr_fn
from models import get_model
from training import save_training_state, get_weights
from physics import get_physics
from settings import DefaultArgParser
from scheduler import get_lr_scheduler

parser = DefaultArgParser()
# NOTE: Some of these arguments should be better tied to their respective class.
parser.add_argument("--method", type=str)
parser.add_argument(
    "--Loss__crop_training_pairs", action=BooleanOptionalAction, default=True
)
parser.add_argument("--Loss__crop_size", type=int, default=48)
parser.add_argument(
    "--ProposedLoss__transforms", type=str, default="Scaling_Transforms"
)
parser.add_argument(
    "--ProposedLoss__stop_gradient", action=BooleanOptionalAction, default=True
)
parser.add_argument("--ProposedLoss__sure_alternative", type=str, default=None)
parser.add_argument("--ProposedLoss__alpha_tradeoff", type=float, default=1.0)
parser.add_argument("--ScalingTransform__kind", type=str, default="padded")
parser.add_argument(
    "--ScalingTransform__antialias", action=BooleanOptionalAction, default=False
)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--memoize_gt", action=BooleanOptionalAction, default=True)
parser.add_argument("--partial_sure", action=BooleanOptionalAction, default=True)
parser.add_argument("--sure_cropped_div", action=BooleanOptionalAction, default=True)
parser.add_argument("--sure_averaged_cst", action=BooleanOptionalAction, default=None)
parser.add_argument("--partial_sure_sr", action=BooleanOptionalAction, default=False)
parser.add_argument("--sure_margin", type=int, default=None)
parser.add_argument("--lr_scheduler_kind", type=str, default="delayed_linear_decay")
parser.add_argument("--optimizer_beta2", type=float, default=0.999)
parser.add_argument(
    "--SyntheticDataset__deterministic_measurements",
    action=BooleanOptionalAction,
    default=True,
)
parser.add_argument("--GroundTruthDataset__split", type=str, default="train")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument("--fine_tuning", action=BooleanOptionalAction, default=False)
parser.add_argument("--fine_tuning_params", action=BooleanOptionalAction, default=None)
args = parser.parse_args()

physics = get_physics(args, device=args.device)

model = get_model(
    args=args,
    physics=physics,
    device=args.device,
)
model.to(args.device)
model.train()

if args.weights is not None:
    weights = get_weights(args.weights, args.device)
    model.load_weights(weights)

loss = get_loss(args=args, physics=physics)

if isdir(args.dataset):
    assert args.fine_tuning, "Datasets of predictors only are only supported for fine-tuning"
    assert args.method == "proposed", "Fine-tuning is only supported for the proposed method"
    from glob import glob
    from os.path import basename
    from torchvision.io import read_image
    from torchvision.transforms import RandomCrop
    dataset = []
    for i, f in enumerate(glob(os.path.join(args.dataset, "*.png"))):
        y = read_image(f)
        y = y.to(args.device)
        y = y.float() / 255.0
        # Discard the alpha channel if it exists
        y = y[:3, :, :]
        # The training algorithm does not use the input image but it is still
        # passed around so we need to create a dummy one
        x = torch.zeros_like(y)
        f_crop = RandomCrop(args.PrepareTrainingPairs__crop_size)
        x, y = f_crop(x), f_crop(y)
        dataset.append((x, y))
else:
    dataset = get_dataset(args=args,
                          purpose="train",
                          physics=physics,
                          device=args.device)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# NOTE: It'd be better to have a more general formula depending on the length
# of the dataset instead of hardcoding these values.
if args.epochs is not None:
    epochs = args.epochs
else:
    if args.dataset == "div2k":
        epochs = 500
    elif args.dataset == "urban100":
        epochs = 4000
    elif args.dataset == "ct":
        epochs = 100
    else:
        epochs = 500

# NOTE: It'd be better if the learning rate was the same for all tasks.
if args.lr is not None:
    lr = args.lr
else:
    if not args.fine_tuning:
        if args.task == "sr":
            lr = 2e-4
        else:
            lr = 5e-4
    else:
        lr = 1e-3

print(f"\nSelected learning rate: {lr:e}\n")

if args.optimizer is not None:
    optimizer_kind = args.optimizer
else:
    if not args.fine_tuning:
        optimizer_kind = "Adam"
    else:
        optimizer_kind = "SGD"

print(f"\nSelected optimizer: {optimizer_kind}\n")

if optimizer_kind == "Adam":
    optimizer_cls = Adam
    optimizer_kwargs = {"betas": (0.9, optimizer_kind_beta2)}
elif optimizer_kind == "SGD":
    optimizer_cls = SGD
    optimizer_kwargs = {}
else:
    raise ValueError(f"Unknown optimizer: {optimizer_kind}")

if args.fine_tuning_params is None:
    params = model.parameters()
else:
    assert args.fine_tuning, "Fine-tuning parameters are only supported for fine-tuning"
    param_keys = [...]
    params = [ model.get_parameter(key) for key in param_keys ]

optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)
scheduler = get_lr_scheduler(
    optimizer=optimizer, epochs=epochs, lr_scheduler_kind=args.lr_scheduler_kind
)

# NOTE: This should be simpler.
if args.checkpoint_interval is not None:
    checkpoint_interval = args.checkpoint_interval
else:
    checkpoint_interval = 50
    if args.dataset == "urban100":
        checkpoint_interval = 400
    elif args.dataset == "ct":
        checkpoint_interval = 50


os.makedirs(args.out_dir, exist_ok=True)
filepath = f"{args.out_dir}/training.csv"
file = open(filepath, "w", newline="", buffering=1)
loss_history_writer = csv.writer(file)
loss_history_writer.writerow(
    [
        "Epoch",
        "Training Loss",
    ]
)


checkpoints_dir = f"{args.out_dir}/checkpoints"
def format_checkpoint_filename(epoch):
    return f"ckp_{epoch:03}.pt"

# Save the initial weights
checkpoint_filename = format_checkpoint_filename(0)
checkpoint_path = f"{checkpoints_dir}/{checkpoint_filename}"
save_training_state(epoch=0,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state_path=checkpoint_path)

# the entire training loop
training_loss_metric = MeanMetric()
for epoch in range(epochs):
    training_loss_metric.reset()

    # a single epoch
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)

        optimizer.zero_grad()

        training_loss = loss(x=x, y=y, model=model)
        training_loss.backward()

        optimizer.step()

        training_loss_metric.update(training_loss.item())

    scheduler.step()

    # log progress to stdout
    epochs_ndigits = len(str(int(epochs)))
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch_training_loss = training_loss_metric.compute().item()
    print(
        f"\t{current_timestamp}\t[{epoch + 1:{epochs_ndigits}d}/{epochs}]\tTraining_Loss: {epoch_training_loss:.2e}"
    )

    # update the training record
    row = [
        epoch + 1,
        epoch_training_loss,
    ]
    loss_history_writer.writerow(row)
    is_final_epoch = epoch + 1 == epochs

    # save the training state regularly and after training completion
    if (epoch % checkpoint_interval == 0) or (epoch == epochs - 1):
        checkpoint_filename = format_checkpoint_filename(epoch + 1)
        checkpoint_path = f"{checkpoints_dir}/{checkpoint_filename}"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
weights = model.get_weights()
torch.save(weights, weights_path)

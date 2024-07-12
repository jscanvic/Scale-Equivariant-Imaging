# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

from argparse import BooleanOptionalAction

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from datetime import datetime
from torchmetrics import MeanMetric

from datasets import get_dataset
from losses import get_loss
from metrics import psnr_fn
from models import get_model
from training import save_training_state
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
args = parser.parse_args()

physics = get_physics(args, device=args.device)

model = get_model(
    args=args,
    physics=physics,
    device=args.device,
)
model.to(args.device)
model.train()

dataset = get_dataset(args=args, purpose="train", physics=physics, device=args.device)

# NOTE: This should be in the loss itself.
if args.partial_sure:
    if args.sure_margin is not None:
        sure_margin = args.sure_margin
    elif args.task == "deblurring":
        assert physics.task == "deblurring"

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

loss = get_loss(args=args, physics=physics, sure_margin=sure_margin)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# NOTE: It'd be better to use approximately the same number of epochs in all
# experiments.
if args.epochs is None:
    epochs = 500 if args.dataset != "ct" else 100
    if args.dataset == "urban100":
        epochs = 4000
else:
    epochs = args.epochs

# NOTE: It'd be better if the learning rate was the same for all tasks.
lr = 2e-4 if args.task == "sr" else 5e-4
optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, args.optimizer_beta2))
scheduler = get_lr_scheduler(
    optimizer=optimizer, epochs=epochs, lr_scheduler_kind=args.lr_scheduler_kind
)

# training loop
training_loss_metric = MeanMetric()
for epoch in range(epochs):
    training_loss_metric.reset()

    # stochastic gradient descent step
    for x, y in dataloader:
        x, y = x.to(args.device), y.to(args.device)

        # NOTE: It'd be great to remove this.
        if args.dataset == "ct":
            assert x.shape[1] == 3
            assert y.shape[1] == 3

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

    # NOTE: This should be simpler.
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
        checkpoint_path = f"{args.out_dir}/checkpoints/ckp_{epoch+1:03}.pt"
        save_training_state(epoch, model, optimizer, scheduler, checkpoint_path)

# save the weights after training completion
weights_path = f"{args.out_dir}/weights.pt"
weights = model.get_weights()
torch.save(weights, weights_path)

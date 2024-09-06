# the library needs to be loaded early on to prevent a crash
# noinspection PyUnresolvedReferences
import bm3d

import torch
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from physics import get_physics
from settings import DefaultArgParser

from glob import glob
from os import makedirs
from os.path import basename

torch.manual_seed(0)
np.random.seed(0)

parser = DefaultArgParser()
parser.add_argument("image_dir", type=str, help="Directory containing images to degrade")
parser.add_argument("out_dir", type=str, help="Directory to save degraded images")
args = parser.parse_args()
physics = get_physics(args, device="cpu")

makedirs(args.out_dir, exist_ok=True)

for image_path in tqdm(glob(f"{args.image_dir}/*.png")):
    x = read_image(image_path)
    x = x.to(torch.float32) / 255
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    elif x.size(0) == 4:
        x = x[:3]
    else:
        assert x.size(0) == 3
    y = physics(x.unsqueeze(0)).squeeze(0)
    out_path = f"{args.out_dir}/{basename(image_path)}"
    save_image(y, out_path)

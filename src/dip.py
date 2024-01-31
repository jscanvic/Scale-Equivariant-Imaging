# code from https://deepinv.github.io/deepinv/auto_examples/basics/demo_dip.html

from torch.nn import Module
from deepinv.models import ConvDecoder, DeepImagePrior


class DIPModel(Module):
    def __init__(self, physics, sr_factor=None):
        super().__init__()
        self.physics = physics
        self.sr_factor = sr_factor


    def forward(self, y):
        img_shape = y.shape[1:]
        if self.sr_factor is not None:
            img_shape = (img_shape[0], int(img_shape[1] * self.sr_factor), int(img_shape[2] * self.sr_factor))

        iterations = 4000
        lr = 5e-3
        channels = 32
        in_size = [16, 16]

        backbone = ConvDecoder(
            img_shape=img_shape, in_size=in_size, channels=channels
        ).to(y.device)

        model = DeepImagePrior(
            backbone,
            learning_rate=lr,
            iterations=iterations,
            input_size=[channels] + in_size,
            verbose=True,
        ).to(y.device)

        return model(y, self.physics)

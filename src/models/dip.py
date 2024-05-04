# code from https://deepinv.github.io/deepinv/auto_examples/basics/demo_dip.html

from torch.nn import Module
from deepinv.models import ConvDecoder, DeepImagePrior as DIP


class DeepImagePrior(Module):
    def __init__(
        self,
        physics,
        sr_factor=None,
        iterations=4000,
        lr=5e-3,
        channels=32,
        in_size=None,
    ):
        super().__init__()
        if in_size is None:
            in_size = [16, 16]
        self.physics = physics
        self.sr_factor = sr_factor
        self.iterations = iterations
        self.lr = lr
        self.channels = channels
        self.in_size = in_size

    def forward(self, y):
        img_shape = y.shape[1:]
        if self.sr_factor is not None:
            img_shape = (
                img_shape[0],
                int(img_shape[1] * self.sr_factor),
                int(img_shape[2] * self.sr_factor),
            )

        backbone = ConvDecoder(
            img_shape=img_shape, in_size=self.in_size, channels=self.channels
        ).to(y.device)

        model = DIP(
            backbone,
            learning_rate=self.lr,
            iterations=self.iterations,
            input_size=[self.channels] + self.in_size,
            verbose=False,
        ).to(y.device)

        return model(y, self.physics)

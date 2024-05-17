from torch.nn import Module
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import TVPrior
from deepinv.optim.optimizers import optim_builder


class TV(Module):
    def __init__(self, physics, lambd, stepsize=1.0, max_iter=300, n_it_max=20, early_stop=True):
        super().__init__()

        self.physics = physics

        self.data_fidelity = L2()
        self.prior = TVPrior(n_it_max=n_it_max)

        self.params_algo = {"stepsize": stepsize, "lambda": lambd}
        self.backbone = optim_builder(
            iteration="PGD",
            prior=self.prior,
            data_fidelity=self.data_fidelity,
            early_stop=early_stop,
            max_iter=max_iter,
            params_algo=self.params_algo,
            verbose=False,
        )

    def forward(self, y):
        return self.backbone(y, self.physics)

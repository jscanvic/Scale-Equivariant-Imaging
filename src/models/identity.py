from torch.nn import Module


class Identity(Module):
    def forward(self, y):
        return y

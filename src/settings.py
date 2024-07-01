import argparse


class DefaultArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--device", type=str, default="cpu")
        self.add_argument("--task", type=str)
        self.add_argument("--kernel", type=str, default=None)
        self.add_argument("--sr_factor", type=int, default=None)
        self.add_argument("--sr_factor", type=int, default=None)
        self.add_argument("--noise_level", type=int, default=5)
        self.add_argument("--dataset", type=str, default="div2k")
        self.add_argument("--resize_gt", action=BooleanOptionalAction, default=True)
        self.add_argument("--gt_size", type=int, default=256)
        self.add_argument("--download", action=BooleanOptionalAction, default=False)

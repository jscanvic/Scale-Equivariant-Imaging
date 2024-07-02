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
        self.add_argument("--datasets_dir", type=str, default="./datasets")
        self.add_voidable_argument("--GroundTruthDataset__size",
                                   "--GroundTruthDataset__no_resize",
                                   type=int,
                                   default=256)
        self.add_argument("--download", action=BooleanOptionalAction, default=False)
        self.add_argument("--model_kind", type=str, default="swinir")


    def add_voidable_argument(self, flag, negative_flag=None, **kwargs):
        if negative_flag is None:
            negative_flag = f"no_{flag}"
        group = self.add_mutually_exclusive_group()
        group.add_argument(flag, **kwargs)
        group.add_argument(negative_flag, action="store_const", const=None, dest=flag)

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
        self.add_argument("--GroundTruthDataset__dataset", type=str, default="div2k")
        self.add_argument("--GroundTruthDataset__datasets_dir", type=str, default="./datasets")
        self.add_argument("--GroundTruthDataset__download", action=BooleanOptionalAction, default=False)
        group = self.mutually_exclusive_group()
        group.add_argument("--GroundTruthDataset__size", type=int, default=256)
        group.add_argument("--GroundTruthDataset__no_resize", action="store_const", dest="GroundTruthDataset__size", const=None)
        self.add_argument("--model_kind", type=str, default="swinir")

from torch.utils.data import Dataset as BaseDataset

from .ground_truth import GroundTruthDataset


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        blueprint,
        device,
        deterministic_measurements,
        unique_seeds,
        physics,
    ):
        super().__init__()
        self.device = device
        self.deterministic_measurements = deterministic_measurements
        self.unique_seeds = unique_seeds
        self.physics_manager = getattr(physics, "__manager")

        self.ground_truth_dataset = GroundTruthDataset(
            blueprint=blueprint,
            **blueprint[GroundTruthDataset.__name__],
        )

    def __getitem__(self, index):
        x = self.ground_truth_dataset[index]
        x = x.to(self.device)

        if self.deterministic_measurements:
            if self.unique_seeds:
                seed = self.ground_truth_dataset.get_unique_id(index)
            else:
                seed = 0
        else:
            seed = None

        x = x.unsqueeze(0)
        y = self.physics_manager.randomly_degrade(x, seed=seed)
        y = y.squeeze(0)
        x = x.squeeze(0)

        return x, y

    def __len__(self):
        return len(self.ground_truth_dataset)

from pathlib import Path

from datasets_loaders.base import BaseDataset
from core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("lolv1")
class LOLv1(BaseDataset):
    def __init__(self, root, split="train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        split_map = {"train": "our485", "test": "eval15"}
        split_folder = split_map[self.split]
        self.low_dir = self.root / split_folder / "low"
        self.high_dir = self.root / split_folder / "high"
        self.filenames = sorted(file.name for file in self.low_dir.glob("*.png"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low = self.load_rgb_tensor(self.low_dir / filename)
        high = self.load_rgb_tensor(self.high_dir / filename)
        low, high = self.apply_transforms([low, high])
        return [low, high]

    def get_meta(self):
        return {
            "name": "lolv1",
            "type": "srgb",
            "paired": True,
            "eval_channel": "y",
        }

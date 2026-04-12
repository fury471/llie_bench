from pathlib import Path

from datasets_loaders.base import BaseDataset
from core.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("lolv2")
class LOLv2(BaseDataset):
    def __init__(self, root, subset="Real_captured", split="Train", transforms=None):
        super().__init__()
        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.transforms = transforms
        self.low_dir = self.root / subset / split / "Low"
        self.normal_dir = self.root / subset / split / "Normal"
        self.filenames = sorted(file.name for file in self.low_dir.glob("*.png"))

    def __len__(self):
        return len(self.filenames)

    def _normal_path_for(self, filename):
        if self.subset == "Real_captured":
            suffix = filename.replace("low", "")
            return self.normal_dir / f"normal{suffix}"
        return self.normal_dir / filename

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low = self.load_rgb_tensor(self.low_dir / filename)
        normal = self.load_rgb_tensor(self._normal_path_for(filename))
        low, normal = self.apply_transforms([low, normal])
        return [low, normal]

    def get_meta(self):
        return {
            "name": f"lolv2_{self.subset.lower()}",
            "type": "srgb",
            "paired": True,
            "eval_channel": "y",
        }

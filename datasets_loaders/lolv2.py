import cv2
import numpy as np
import torch
from pathlib import Path
from datasets_loaders.base import BaseDataset
from core.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register("lolv2")
class LOLv2(BaseDataset):
    def __init__(self, root, subset="Real_captured", split="Train"):
        super().__init__()
        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.low_dir = self.root / subset / split / "Low"
        self.normal_dir = self.root / subset / split / "Normal"
        self.filenames = sorted([f.name for f in self.low_dir.glob("*.png")])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get the filename and build full paths for both low and high images
        fname = self.filenames[idx]
        low_path = self.low_dir / fname

        if self.subset == "Real_captured":
            number = fname.replace("low", "")
            normal_path = self.normal_dir / f"normal{number}"
        else:
            normal_path = self.normal_dir / fname

        # Load both images using OpenCV
        low = cv2.imread(str(low_path))
        normal = cv2.imread(str(normal_path))

        # Convert from BGR to RGB and normalize to [0, 1]
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        low = low.astype(np.float32) / 255.0
        normal = normal.astype(np.float32) / 255.0

        # Convert to PyTorch tensors and permute from (H, W, C) to (C, H, W)
        low = torch.from_numpy(low).permute(2, 0, 1)
        normal = torch.from_numpy(normal).permute(2, 0, 1)

        return [low, normal]
    
    def get_meta(self):
        return {
            "name": f"lolv2_{self.subset.lower()}",
            "type": "srgb",
            "paired": True,
            "eval_channel": "y"
        }
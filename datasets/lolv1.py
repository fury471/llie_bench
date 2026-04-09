import cv2
import numpy as np
import torch
from pathlib import Path
from datasets.base import BaseDataset
from core.registry import DATASET_REGISTRY

class LOLv1(BaseDataset):
    def __init__(self, root, split="train"):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.low_dir = self.root / split / "low"
        self.high_dir = self.root / split / "high"
        self.filenames = sorted([f.name for f in self.low_dir.glob("*.png")])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get the filename and build full paths for both low and high images
        fname = self.filenames[idx]
        low_path = self.low_dir / fname
        high_path = self.high_dir / fname

        # Load both images using OpenCV
        low = cv2.imread(str(low_path))
        high = cv2.imread(str(high_path))

        # Convert from BGR to RGB and normalize to [0, 1]
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)
        low = low.astype(np.float32) / 255.0
        high = high.astype(np.float32) / 255.0

        # Convert to PyTorch tensors and permute from (H, W, C) to (C, H, W)
        low = torch.from_numpy(low).permute(2, 0, 1)
        high = torch.from_numpy(high).permute(2, 0, 1)

        return [low, high]
    
    def get_meta(self):
        return {
            "name": "lol_v1",
            "type": "srgb",
            "paired": True,
            "eval_channel": "y"
        }
    
# Register the dataset in the global registry
DATASET_REGISTRY["lolv1"] = LOLv1
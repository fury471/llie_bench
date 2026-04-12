from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """Abstract base class for all datasets."""

    def __init__(self):
        super().__init__()
        self.transforms = None

    @staticmethod
    def load_rgb_tensor(path):
        """Load an RGB image from disk and return a normalized CHW tensor."""
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.astype(np.float32) / 255.0)
        return torch.from_numpy(image).permute(2, 0, 1)

    def apply_transforms(self, tensors):
        """Apply optional paired transforms to a list of tensors."""
        if self.transforms is None:
            return tensors
        return self.transforms(tensors)

    @abstractmethod
    def __getitem__(self, idx):
        """Return one sample from the dataset by index."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the total number of samples."""
        pass

    @abstractmethod
    def get_meta(self):
        """Return metadata with keys like name, type, paired, and eval_channel."""
        pass

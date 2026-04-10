from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(ABC, Dataset):
    """Abstract base class for all datasets."""

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
        """Return metadata dict with keys: name, type (srgb/raw), paired (bool), eval_channel."""
        pass
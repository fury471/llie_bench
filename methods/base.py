from abc import ABC, abstractmethod

class BaseMethod(ABC):
    """Abstract base class for all methods."""
    @abstractmethod
    def forward(self, batch):
        """Run the method on a batch of data."""
        pass

    @abstractmethod
    def compute_loss(self, batch):
        """Compute the loss for a batch of data."""
        pass

    @abstractmethod
    def load_ckpt(self, path):
        """Load model weights from a checkpoint."""
        pass

    @abstractmethod
    def get_meta(self):
        """Return metadata dict with keys: name, type (srgb/raw), paired (bool)."""
        pass

from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    @abstractmethod
    def compute(self, pred, gt):
        """Compute the metric given predictions and ground truth."""
        pass

    @abstractmethod
    def aggregate(self):
        """Aggregate results across batches if needed."""
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state for a new evaluation."""
        pass
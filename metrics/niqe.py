import pyiqa
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register("niqe")
class NIQEMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []
        self.niqe = pyiqa.create_metric('niqe', as_loss=False)


    def compute(self, pred, gt=None):
        # pyiqa expects inputs to be in the [0, 1] range with shape (B, C, H, W).
        # We unsqueeze to add the Batch dimension.
        score = self.niqe(pred.unsqueeze(0))
        self.values.append(score.item())
        return score
    
    def aggregate(self):
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []

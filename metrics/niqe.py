import pyiqa

from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("niqe")
class NIQEMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []
        self.niqe = None
        self.device = None

    def _get_metric(self, device):
        if self.niqe is None:
            self.niqe = pyiqa.create_metric("niqe", as_loss=False)
        if self.device != device:
            self.niqe = self.niqe.to(device)
            self.device = device
        return self.niqe

    def compute(self, pred, gt=None):
        niqe_metric = self._get_metric(pred.device)
        score = niqe_metric(pred.unsqueeze(0))
        self.values.append(score.item())
        return score

    def aggregate(self):
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []

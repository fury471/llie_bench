from piqa import SSIM
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY

class SSIMMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self, pred, gt):
        ssim_module = SSIM()
        ssim = ssim_module(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        self.values.append(ssim)
        return ssim
    
    def aggregate(self):
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []

# Register the metric in the global registry
METRIC_REGISTRY["ssim"] = SSIMMetric
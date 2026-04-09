import torch
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY

class PSNRMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        self.values.append(psnr.item())
        return psnr
    
    def aggregate(self):
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []

# Register the metric in the global registry
METRIC_REGISTRY["psnr"] = PSNRMetric
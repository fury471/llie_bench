import torch
from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register("psnr")
class PSNRMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []

    def compute(self, pred, gt):
        mse = torch.mean((pred - gt) ** 2).clamp_min(1e-12)
        psnr = 10 * torch.log10(1.0 / mse)
        self.values.append(psnr.item())
        return psnr
    
    def aggregate(self):
        return sum(self.values) / len(self.values)
    
    def reset(self):
        self.values = []

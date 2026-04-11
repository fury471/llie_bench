from piqa import SSIM

from metrics.base import BaseMetric
from core.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("ssim")
class SSIMMetric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.values = []
        self.ssim_module = None
        self.device = None

    def _get_metric(self, device):
        if self.ssim_module is None:
            self.ssim_module = SSIM()
        if self.device != device:
            self.ssim_module = self.ssim_module.to(device)
            self.device = device
        return self.ssim_module

    def compute(self, pred, gt):
        ssim_module = self._get_metric(pred.device)
        ssim = ssim_module(pred.unsqueeze(0), gt.unsqueeze(0)).item()
        self.values.append(ssim)
        return ssim

    def aggregate(self):
        return sum(self.values) / len(self.values)

    def reset(self):
        self.values = []

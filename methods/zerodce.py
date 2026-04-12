import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

"""
+------------------------------------------------------------------+
|                          Zero-DCE Losses                         |
+------------------------------------------------------------------+
| Exposure                : brightness target regularization       |
| Spatial Consistency     : preserve local structure               |
| Color Constancy         : keep RGB channels balanced             |
| Illumination Smoothness : smooth curve parameters spatially      |
+------------------------------------------------------------------+
"""


@METHOD_REGISTRY.register("zerodce")
class ZeroDCE(BaseMethod):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 24, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, batch):
        x = batch[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        return self.tanh(self.conv7(x))

    def exposure_loss(self, enhanced, patch_size=16, target=0.6):
        """Penalize patches that deviate from the target brightness."""
        pooled = F.avg_pool2d(enhanced, patch_size)
        return torch.mean((pooled - target) ** 2)

    def spatial_consistency_loss(self, enhanced, original):
        """Keep local structure similar to the original image."""
        enhanced_dx = enhanced[:, :, :, :-1] - enhanced[:, :, :, 1:]
        original_dx = original[:, :, :, :-1] - original[:, :, :, 1:]
        enhanced_dy = enhanced[:, :, :-1, :] - enhanced[:, :, 1:, :]
        original_dy = original[:, :, :-1, :] - original[:, :, 1:, :]
        return torch.mean((enhanced_dx - original_dx) ** 2) + torch.mean((enhanced_dy - original_dy) ** 2)

    def color_constancy_loss(self, enhanced):
        """Keep RGB channel statistics balanced."""
        mean_r = torch.mean(enhanced[:, 0])
        mean_g = torch.mean(enhanced[:, 1])
        mean_b = torch.mean(enhanced[:, 2])
        return (mean_r - mean_g) ** 2 + (mean_g - mean_b) ** 2 + (mean_b - mean_r) ** 2

    def illumination_smoothness_loss(self, curve_params):
        """Keep curve parameters spatially smooth."""
        grad_x = torch.abs(curve_params[:, :, :, :-1] - curve_params[:, :, :, 1:])
        grad_y = torch.abs(curve_params[:, :, :-1, :] - curve_params[:, :, 1:, :])
        return torch.mean(grad_x) + torch.mean(grad_y)

    def compute_loss(self, batch):
        """Compute the full Zero-DCE objective."""
        original = batch[0]
        curve_params = self.forward(batch)
        enhanced = self.enhance(batch)

        loss_exp = self.exposure_loss(enhanced)
        loss_spa = self.spatial_consistency_loss(enhanced, original)
        loss_col = self.color_constancy_loss(enhanced)
        loss_ill = self.illumination_smoothness_loss(curve_params)

        return 10 * loss_spa + 5 * loss_exp + loss_col + 20 * loss_ill

    def load_ckpt(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict)

    def get_meta(self):
        return {
            "name": "zerodce",
            "type": "srgb",
            "paired": False,
            "description": "Zero-Reference Deep Curve Estimation",
            "paper_url": "https://arxiv.org/abs/2001.06826",
        }

    def enhance(self, batch):
        x = batch[0]
        curve_params = self.forward(batch)
        enhanced = x.clone()
        for index in range(8):
            curves = curve_params[:, index * 3:(index + 1) * 3]
            enhanced = enhanced + curves * enhanced * (1.0 - enhanced)
        return enhanced

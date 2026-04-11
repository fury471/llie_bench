import torch
import torch.nn as nn
from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

"""
+------------------------------------------------------------------+
|                       Enhancement Controls                       |
+------------------------------------------------------------------+
| Exposure                : controls overall brightness target     |
| Spatial Consistency     : keeps structure similar to input       |
| Color Constancy         : keeps colors natural                   |
| Illumination Smoothness : makes enhancement smooth spatially     |
+------------------------------------------------------------------+
"""
@METHOD_REGISTRY.register("zerodce")
class ZeroDCE(BaseMethod):
    def __init__(self):
        super().__init__()
        # define the model architecture
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # input: 3 channels (RGB), output: 32 channels, kernel size: 3
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 24, 3, padding=1)
        # activations
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, batch):
        # The input batch is a list — batch[0] is the low-light image.
        x = batch[0]
        # pass through the model layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.tanh(self.conv7(x))
        return x
    
    def exposure_loss(self, enhanced, patch_size=16, target=0.6):
        """Penalize patches that deviate from target brightness."""
        pool = torch.nn.functional.avg_pool2d(enhanced, patch_size)
        loss = torch.mean((pool - target) ** 2)
        return loss
    
    def spatial_consistency_loss(self, enhanced, original):
        """Keep spatial structure similar to the original image."""
        # enhanced and original are both (B, 3, H, W)
        # compute differences with neighbors in 4 directions
        left = enhanced[:, :, :, :-1] - enhanced[:, :, :, 1:]
        right = original[:, :, :, :-1] - original[:, :, :, 1:]
        top = enhanced[:, :, :-1, :] - enhanced[:, :, 1:, :]
        bottom = original[:, :, :-1, :] - original[:, :, 1:, :]

        loss = torch.mean((left - right) ** 2) + torch.mean((top - bottom) ** 2)
        return loss
    
    def color_constancy_loss(self, enhanced):
        """Keep color channels balanced."""
        # enhanced and original are both (B, 3, H, W)
        mean_r = torch.mean(enhanced[:, 0, :, :])
        mean_g = torch.mean(enhanced[:, 1, :, :])
        mean_b = torch.mean(enhanced[:, 2, :, :])

        loss = (mean_r - mean_g) ** 2 + (mean_g - mean_b) ** 2 + (mean_b - mean_r) ** 2
        return loss
    
    def illumination_smoothness_loss(self, curve_params):
        """Keep curve parameters spatially smooth."""
        # curve_params is (B, 24, H, W)
        grad_x = torch.abs(curve_params[:, :, :, :-1] - curve_params[:, :, :, 1:])
        grad_y = torch.abs(curve_params[:, :, :-1, :] - curve_params[:, :, 1:, :])
        loss = torch.mean(grad_x) + torch.mean(grad_y)
        return loss

    def compute_loss(self, batch):
        """Compute the full Zero-DCE loss with all 4 components."""
        x = batch[0]
        curve_params = self.forward(batch)
        enhanced = self.enhance(batch)

        loss_exp = self.exposure_loss(enhanced)
        loss_spa = self.spatial_consistency_loss(enhanced, x)
        loss_col = self.color_constancy_loss(enhanced)
        loss_ill = self.illumination_smoothness_loss(curve_params)

        # weighted sum — weights from the original paper
        total_loss = (
            10 * loss_spa +
            5  * loss_exp +
            1  * loss_col +
            20 * loss_ill
        )
        return total_loss

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model_state_dict"])

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
        for i in range(8):
            A = curve_params[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + A * enhanced * (1.0 - enhanced)
        return enhanced


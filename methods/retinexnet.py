import torch
import torch.nn as nn
from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

"""
+------------------------------------------------------------------+
|                      RetinexNet Architecture                     |
+------------------------------------------------------------------+
| DecomNet    : decomposes image into reflectance + illumination   |
| EnhanceNet  : enhances the illumination map                      |
| Final output: enhanced_illumination × reflectance                |
+------------------------------------------------------------------+
"""

class DecomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # define decomnet architecture
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 4, 3, padding=1)
        # activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.sigmoid(self.conv5(x))
        # x shape is (B, 4, H, W) — split into reflectance and illumination
        R = x[:, :3, :, :]
        I = x[:, 3:, :, :]
        return R, I
    
class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)
        # activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    
@METHOD_REGISTRY.register("retinexnet")
class RetinexNet(BaseMethod):
    def __init__(self):
        super().__init__()
        self.decom_net = DecomNet()
        self.enhance_net = EnhanceNet()

    def forward(self, batch):
        # get low-light image from batch[0]
        x = batch[0]
        # pass through DecomNet → get R and I
        R, I = self.decom_net(x)
        # pass through EnhanceNet → get enhanced illumination I_enhanced
        I_enhanced = self.enhance_net(I)
        # combine: enhanced = R * I_enhanced
        enhanced = R * I_enhanced
        return enhanced
    
    def enhance(self, batch):
        return self.forward(batch)

    def compute_loss(self, batch):
        low = batch[0]
        high = batch[1]
        enhanced = self.forward(batch)
        loss = torch.nn.functional.l1_loss(enhanced, high)
        return loss
    
    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model_state_dict"])

    def get_meta(self):
        return {
            "name": "retinexnet",
            "type": "srgb",
            "paired": True,
            "description": "Deep Retinex Decomposition for Low-Light Enhancement",
            "paper_url": "https://arxiv.org/abs/1808.04560",
        }
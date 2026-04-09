import torch
import torch.nn as nn
from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY

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
    
    def compute_loss(self, batch):
        enhanced = self.forward(batch)
        # compute average brightness of the enhanced image
        avg_brightness = torch.mean(enhanced)
        # target brightness is 0.6
        target_brightness = 0.6
        exposure_loss = torch.mean((avg_brightness - target_brightness) ** 2)
        return exposure_loss

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
    
# Register the method in the global registry
METHOD_REGISTRY["zerodce"] = ZeroDCE


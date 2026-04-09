from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY
import cv2
import numpy as np
import torch

class CLAHE(BaseMethod):
    def __init__(self, clip_limit=2.0, tile_size=8):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    
    def forward(self, batch):
        # The input batch is a list — batch[0] is a tensor of shape (B, 3, H, W) with values in [0, 1].
        """Apply CLAHE enhancement to a batch of images."""
        # get the input tensor from batch[0] and move to CPU and convert to numpy
        x = batch[0]
        imgs = x.cpu().numpy()

        enhanced = []
        # loop over each image in the batch
        for img in imgs:
            # (3, H, W) → (H, W, 3) and scale to uint8
            img_hwc = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            # RGB → LAB
            img_lab = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2LAB)

            # apply CLAHE on L channel only
            img_lab[:, :, 0] = self.clahe.apply(img_lab[:, :, 0])

            # LAB → RGB
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

            # normalize to [0, 1] and (H, W, 3) → (3, H, W)
            img_chw = img_rgb.astype(np.float32) / 255.0
            enhanced.append(img_chw.transpose(2, 0, 1))
        
        return torch.from_numpy(np.stack(enhanced))
    
    def compute_loss(self, batch):
        raise NotImplementedError("Traditional methods do not support training.")

    def load_ckpt(self, path):
        pass  # no weights to load

    def get_meta(self):
        return {
            "name": "clahe",
            "type": "srgb",
            "paired": False,
            "description": "Contrast Limited Adaptive Histogram Equalization (CLAHE)",
            "paper_url": "https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE_(CLAHE)",
        }
    
    def enhance(self, batch):
        return self.forward(batch)

# Register the method in the global registry
METHOD_REGISTRY["clahe"] = CLAHE
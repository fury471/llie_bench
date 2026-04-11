import cv2
import numpy as np
import torch

from methods.base import BaseMethod
from core.registry import METHOD_REGISTRY


@METHOD_REGISTRY.register("clahe")
class CLAHE(BaseMethod):
    def __init__(self, clip_limit=2.0, tile_size=8):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    def forward(self, batch):
        """Apply CLAHE enhancement to a batch of images."""
        x = batch[0]
        device = x.device
        imgs = x.detach().cpu().numpy()

        enhanced = []
        for img in imgs:
            img_hwc = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            img_lab = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2LAB)
            img_lab[:, :, 0] = self.clahe.apply(img_lab[:, :, 0])
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            img_chw = img_rgb.astype(np.float32) / 255.0
            enhanced.append(img_chw.transpose(2, 0, 1))

        return torch.from_numpy(np.stack(enhanced)).to(device)

    def compute_loss(self, batch):
        raise NotImplementedError("Traditional methods do not support training.")

    def load_ckpt(self, path):
        pass

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

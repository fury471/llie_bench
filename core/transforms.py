import random

import torch

class RandomCrop:
    """Randomly crop a patch from a list of image tensors."""
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, tensors):
        # tensors is a list of (C, H, W) tensors
        _, h, w = tensors[0].shape
        if self.patch_size > h or self.patch_size > w:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than image size {(h, w)}."
            )
        rh = random.randint(0, h - self.patch_size)
        rw = random.randint(0, w - self.patch_size)
        return [t[:, rh:rh+self.patch_size, rw:rw+self.patch_size] for t in tensors]
    
class RandomFlip:
    """Randomly flip a list of image tensors horizontally."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensors):
        if random.random() < self.p:
            return [torch.flip(t, dims=[2]) for t in tensors]  # flip width dimension
        return tensors
    
class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensors):
        for t in self.transforms:
            tensors = t(tensors)
        return tensors


def build_transforms(config):
    """Build a composed transform pipeline from a config dictionary."""
    transforms = []
    if config.get("patch_size"):
        transforms.append(RandomCrop(config["patch_size"]))
    if config.get("random_flip", False):
        transforms.append(RandomFlip())
    return Compose(transforms) if transforms else None

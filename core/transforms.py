import torch
import random

class RandomCrop:
    """Randomly crop a patch from a list of image tensors."""
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, tensors):
        # tensors is a list of (C, H, W) tensors
        _, h, w = tensors[0].shape
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
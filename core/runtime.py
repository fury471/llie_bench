from pathlib import Path

import numpy as np
import torch
import cv2

from core.checkpoint import resolve_checkpoint_path


def resolve_device():
    """Return the preferred runtime device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def move_batch_to_device(batch, device):
    """Move a list-like batch of tensors to the target device."""
    return [tensor.to(device) for tensor in batch]


def method_requires_checkpoint(method):
    """Treat methods with trainable parameters as checkpoint-based methods."""
    return any(parameter.numel() > 0 for parameter in method.parameters())


def load_method_checkpoint(method, method_name, ckpt=None, ckpt_dir=None, required=False):
    """
    Resolve and load a checkpoint for a method.

    Returns a tuple of (checkpoint_path_or_none, status_message).
    """
    ckpt_path = resolve_checkpoint_path(ckpt, ckpt_dir)

    if ckpt_path is None:
        if required:
            raise ValueError(
                f"Method '{method_name}' requires a checkpoint. "
                "Please provide a valid .pth checkpoint file or checkpoint directory."
            )
        return None, f"No checkpoint for method '{method_name}'; using current weights."

    try:
        method.load_ckpt(str(ckpt_path))
    except (RuntimeError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Failed to load checkpoint '{ckpt_path}' for method '{method_name}'. "
            "The checkpoint may belong to a different method or incompatible architecture."
        ) from exc

    return ckpt_path, f"Loaded checkpoint: {ckpt_path}"


def rgb_image_to_tensor(image, device):
    """Convert an RGB uint8/float numpy image to a BCHW float tensor on the target device."""
    image = np.ascontiguousarray(image.astype(np.float32) / 255.0)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_rgb_image(tensor):
    """Convert a BCHW/CHW tensor in [0, 1] to an RGB uint8 numpy image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def resize_if_needed(image, max_side):
    """Resize image so longest side <= max_side. Returns (image, message|None)."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, None
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, f"Resized from {w}×{h} to {new_w}×{new_h}."


def is_oom(exc):
    """Check if exception is an out-of-memory error."""
    msg = str(exc).lower()
    return "out of memory" in msg or "not enough memory" in msg

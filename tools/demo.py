from pathlib import Path

import gradio as gr
import numpy as np
import torch

import plugins
from core.checkpoint import resolve_checkpoint_path
from core.registry import METHOD_REGISTRY, lookup


def method_requires_checkpoint(method):
    """Treat methods with trainable parameters as checkpoint-based methods."""
    return any(param.requires_grad for param in method.parameters())


def resolve_demo_checkpoint(method, method_name, path_or_dir):
    """Accept either a checkpoint file path or a checkpoint directory."""
    requires_ckpt = method_requires_checkpoint(method)

    if not path_or_dir:
        if requires_ckpt:
            raise gr.Error(
                f"Method '{method_name}' requires a checkpoint. "
                "Please provide a .pth checkpoint file or a checkpoint directory."
            )
        return None, f"Method '{method_name}' does not need a checkpoint."

    if not requires_ckpt:
        return None, (
            f"Method '{method_name}' does not need a checkpoint. "
            "Ignoring the provided checkpoint path."
        )

    candidate = Path(path_or_dir)
    if candidate.suffix.lower() == ".pth":
        if not candidate.is_file():
            raise gr.Error(f"Checkpoint file not found: {candidate}")
        resolved = resolve_checkpoint_path(str(candidate), None)
        return resolved, f"Loaded checkpoint file: {resolved}"

    if candidate.exists() and not candidate.is_dir():
        raise gr.Error(f"Expected a checkpoint file or directory, got: {candidate}")

    if not candidate.exists():
        raise gr.Error(f"Checkpoint path not found: {candidate}")

    resolved = resolve_checkpoint_path(None, str(candidate))
    if resolved is None:
        raise gr.Error(
            f"No .pth checkpoint found in directory: {candidate}. "
            "Please point to a valid checkpoint file or directory."
        )
    return resolved, f"Loaded latest checkpoint from directory: {resolved}"


def enhance(image, method_name, ckpt_path_or_dir):
    method = lookup(METHOD_REGISTRY, method_name)()
    ckpt_path, status = resolve_demo_checkpoint(method, method_name, ckpt_path_or_dir)
    if ckpt_path:
        method.load_ckpt(str(ckpt_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = method.to(device)
    method.eval()

    img = image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = method.enhance([img_tensor])

    output = enhanced.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output, status


method_choices = sorted(METHOD_REGISTRY.keys())
default_method = "zerodce" if "zerodce" in METHOD_REGISTRY else method_choices[0]

demo = gr.Interface(
    fn=enhance,
    inputs=[
        gr.Image(type="numpy", label="Input Dark Image"),
        gr.Dropdown(choices=method_choices, value=default_method, label="Method"),
        gr.Textbox(value="", label="Checkpoint Path or Directory (optional)"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Enhanced Image"),
        gr.Textbox(label="Status"),
    ],
    title="LLIE Bench Demo",
    description="Upload a low-light image and select a method to enhance it.",
)

if __name__ == "__main__":
    demo.launch()

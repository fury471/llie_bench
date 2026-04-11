from pathlib import Path

import gradio as gr
import cv2
import numpy as np
import torch

import plugins
from core.checkpoint import resolve_checkpoint_path
from core.registry import METHOD_REGISTRY, lookup


def method_requires_checkpoint(method):
    """Treat methods with trainable parameters as checkpoint-based methods."""
    return any(param.requires_grad for param in method.parameters())


def resize_for_demo(image, max_side):
    """Resize an image to a maximum side length while preserving aspect ratio."""
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image, None

    scale = max_side / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    message = f"Input resized from {width}x{height} to {new_width}x{new_height} for demo inference."
    return resized, message


def build_demo_image_candidates(image, max_sides=(1600, 1280, 1024, 768)):
    """Try larger resolutions first, then only downscale further if memory requires it."""
    height, width = image.shape[:2]
    longest_side = max(height, width)

    candidates = []
    seen_shapes = set()

    if longest_side <= max_sides[0]:
        candidates.append((image, None))
        seen_shapes.add((height, width))

    for max_side in max_sides:
        resized, message = resize_for_demo(image, max_side)
        resized_shape = resized.shape[:2]
        if resized_shape not in seen_shapes:
            candidates.append((resized, message))
            seen_shapes.add(resized_shape)

    return candidates


def is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "not enough memory" in message


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
        try:
            method.load_ckpt(str(ckpt_path))
        except RuntimeError as exc:
            raise gr.Error(
                f"Failed to load checkpoint '{ckpt_path}' for method '{method_name}'. "
                "The checkpoint may belong to a different method or incompatible architecture.\n\n"
                f"Details: {exc}"
            ) from exc

    image_candidates = build_demo_image_candidates(image)
    status_parts = [status]

    def run_on_device(image_array, target_device):
        local_method = method.to(target_device)
        local_method.eval()
        img = image_array.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        local_tensor = input_tensor.to(target_device)
        with torch.no_grad():
            return local_method.enhance([local_tensor])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhanced = None
    resize_status = None
    last_exc = None

    for candidate_image, candidate_status in image_candidates:
        try:
            enhanced = run_on_device(candidate_image, device)
            resize_status = candidate_status
            break
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if device == "cuda" and is_oom_error(exc):
                last_exc = exc
                method = method.cpu()
                torch.cuda.empty_cache()
                continue
            raise gr.Error(f"Inference failed on {device}: {exc}") from exc

    if enhanced is None and device == "cuda":
        status_parts.append("CUDA ran out of memory, so inference retried on CPU.")
        for candidate_image, candidate_status in image_candidates:
            try:
                enhanced = run_on_device(candidate_image, "cpu")
                resize_status = candidate_status
                break
            except (torch.OutOfMemoryError, RuntimeError) as cpu_exc:
                if is_oom_error(cpu_exc):
                    last_exc = cpu_exc
                    continue
                raise gr.Error(f"Inference failed on cpu: {cpu_exc}") from cpu_exc

    if enhanced is None:
        raise gr.Error(
            "Inference still ran out of memory after trying progressively smaller demo sizes. "
            "Please use a smaller image for the demo or resize the image before uploading.\n\n"
            f"Details: {last_exc}"
        )

    if resize_status:
        status_parts.append(resize_status)

    output = enhanced.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output, " ".join(status_parts)


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

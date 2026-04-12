"""
LLIE Bench web application.

Usage:
    python -m tools.app
    python -m tools.app --share
    python -m tools.app --port 8080
    python -m tools.app --host 0.0.0.0
"""

import argparse
import logging
from pathlib import Path

import gradio as gr
import torch

import plugins
from core.registry import METHOD_REGISTRY, lookup
from core.runtime import (
    is_oom,
    load_method_checkpoint,
    method_requires_checkpoint,
    resolve_device,
    resize_if_needed,
    rgb_image_to_tensor,
    tensor_to_rgb_image,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llie_bench.app")

RESIZE_CANDIDATES = [1600, 1280, 1024, 768, 512]
MAX_ORIGINAL_SIDE = 4096


def build_image_candidates(image):
    """
    Build a list of (image_array, resize_message_or_none) from largest to smallest.
    """
    height, width = image.shape[:2]
    seen_shapes = set()
    candidates = []

    if max(height, width) <= MAX_ORIGINAL_SIDE:
        candidates.append((image, None))
        seen_shapes.add((height, width))

    for max_side in RESIZE_CANDIDATES:
        resized, message = resize_if_needed(image, max_side)
        resized_shape = resized.shape[:2]
        if resized_shape not in seen_shapes:
            candidates.append((resized, message))
            seen_shapes.add(resized_shape)

    return candidates


def run_single(method, image_array, device):
    """Run one inference pass for one image on one device."""
    local_method = method.to(device)
    local_method.eval()
    tensor = rgb_image_to_tensor(image_array, device)
    with torch.no_grad():
        return local_method.enhance([tensor])


def infer_with_fallback(method, image, primary_device):
    """
    Try progressively smaller resolutions on the primary device, then retry on CPU.
    """
    candidates = build_image_candidates(image)
    status_parts = []

    for resized, resize_message in candidates:
        try:
            output = run_single(method, resized, primary_device)
            if resize_message:
                status_parts.append(resize_message)
            return output, status_parts
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not is_oom(exc):
                raise gr.Error(f"Inference failed on {primary_device}: {exc}") from exc
            logger.warning("OOM on %s at image size %s", primary_device, resized.shape[:2])
            if primary_device == "cuda":
                torch.cuda.empty_cache()

    if primary_device == "cuda":
        status_parts.append("CUDA ran out of memory, retrying on CPU.")
        method = method.cpu()
        torch.cuda.empty_cache()
        for resized, resize_message in candidates:
            try:
                output = run_single(method, resized, "cpu")
                if resize_message:
                    status_parts.append(resize_message)
                return output, status_parts
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not is_oom(exc):
                    raise gr.Error(f"CPU inference failed: {exc}") from exc
                logger.warning("OOM on CPU at image size %s", resized.shape[:2])

    raise gr.Error(
        "Ran out of memory at all attempted resolutions. "
        f"Tried sizes: {RESIZE_CANDIDATES}. "
        "Please upload a smaller image."
    )


def resolve_app_checkpoint(method, method_name, checkpoint_input):
    """Resolve checkpoint input for the web app and return a user-facing status message."""
    requires_checkpoint = method_requires_checkpoint(method)
    checkpoint_text = checkpoint_input.strip() if checkpoint_input else None

    if not requires_checkpoint:
        if checkpoint_text:
            return (
                f"Method '{method_name}' does not need a checkpoint. "
                "Ignoring the provided checkpoint path."
            )
        return f"Method '{method_name}' does not need a checkpoint."

    checkpoint_file = checkpoint_text
    checkpoint_dir = None
    if checkpoint_file and Path(checkpoint_file).is_dir():
        checkpoint_dir = checkpoint_file
        checkpoint_file = None

    try:
        _, status = load_method_checkpoint(
            method,
            method_name,
            ckpt=checkpoint_file,
            ckpt_dir=checkpoint_dir,
            required=True,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return status


def enhance(image, method_name, checkpoint_input):
    """
    Gradio handler: validate inputs, load method, run inference.
    Returns (enhanced_image, status_string).
    """
    if image is None:
        raise gr.Error("Please upload an image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise gr.Error("Expected an RGB image with 3 channels.")

    logger.info("Request: method=%s ckpt='%s' image=%s", method_name, checkpoint_input, image.shape)

    try:
        method = lookup(METHOD_REGISTRY, method_name)()
    except ValueError as exc:
        raise gr.Error(f"Unknown method '{method_name}': {exc}") from exc

    checkpoint_status = resolve_app_checkpoint(method, method_name, checkpoint_input)
    logger.info("Checkpoint status: %s", checkpoint_status)

    device = resolve_device()
    logger.info("Running inference on: %s", device)

    output_tensor, inference_status = infer_with_fallback(method, image, device)
    output_image = tensor_to_rgb_image(output_tensor)

    status_parts = [checkpoint_status] + inference_status
    final_status = " | ".join(part for part in status_parts if part)
    logger.info("Done. %s", final_status)

    return output_image, final_status


def build_interface():
    method_choices = sorted(METHOD_REGISTRY.keys())
    default_method = "zerodce" if "zerodce" in METHOD_REGISTRY else method_choices[0]

    return gr.Interface(
        fn=enhance,
        inputs=[
            gr.Image(type="numpy", label="Input Low-Light Image"),
            gr.Dropdown(
                choices=method_choices,
                value=default_method,
                label="Enhancement Method",
            ),
            gr.Textbox(
                value="",
                placeholder=(
                    "e.g. checkpoints/zerodce_lolv1 or "
                    "checkpoints/zerodce_lolv1/checkpoint_epoch_099.pth"
                ),
                label="Checkpoint Path or Directory (required for learned methods)",
            ),
        ],
        outputs=[
            gr.Image(type="numpy", label="Enhanced Image"),
            gr.Textbox(label="Status"),
        ],
        title="LLIE Bench",
        description=(
            "Low-light image enhancement benchmark web app. "
            "Upload a dark image, select a method, and provide a checkpoint for learned methods."
        ),
        flagging_mode="never",
    )


def main():
    parser = argparse.ArgumentParser(description="LLIE Bench web app")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    args = parser.parse_args()

    logger.info("Starting LLIE Bench on %s:%d (share=%s)", args.host, args.port, args.share)

    app = build_interface()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

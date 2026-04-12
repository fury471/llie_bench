import argparse
from pathlib import Path

import cv2
import torch

import plugins
from core.runtime import (
    load_method_checkpoint,
    method_requires_checkpoint,
    resolve_device,
    rgb_image_to_tensor,
    tensor_to_rgb_image,
)
from core.registry import METHOD_REGISTRY, lookup


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--method", type=str, required=True, help="Method name like zerodce or clahe")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint file")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to auto-load the latest checkpoint from")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, default="results/enhanced.png", help="Path to save the output image")
    args = parser.parse_args()

    method = lookup(METHOD_REGISTRY, args.method)()
    if method_requires_checkpoint(method):
        try:
            _, status = load_method_checkpoint(
                method,
                args.method,
                ckpt=args.ckpt,
                ckpt_dir=args.ckpt_dir,
                required=True,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(status)
    else:
        print(f"Method '{args.method}' does not need a checkpoint.")

    device = resolve_device()
    method = method.to(device)
    method.eval()

    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read input image: {args.input}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = rgb_image_to_tensor(image_rgb, device)

    with torch.no_grad():
        enhanced = method.enhance([image_tensor])

    output_rgb = tensor_to_rgb_image(enhanced)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output_bgr)
    print(f"Enhanced image saved to {output_path}")


if __name__ == "__main__":
    main()

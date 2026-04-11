import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from core.checkpoint import resolve_checkpoint_path
from core.registry import METHOD_REGISTRY, lookup

# import plugins so they register themselves
import plugins


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--method", type=str, required=True, help="Method name e.g. zerodce, clahe")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to auto-load latest checkpoint from")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="results/enhanced.png", help="Path to save output")
    args = parser.parse_args()

    method = lookup(METHOD_REGISTRY, args.method)()
    ckpt_path = resolve_checkpoint_path(args.ckpt, args.ckpt_dir)
    if ckpt_path:
        print(f"Auto-loading checkpoint: {ckpt_path}")
        method.load_ckpt(str(ckpt_path))
    else:
        print("No checkpoint; using random weights")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = method.to(device)
    method.eval()

    img = cv2.imread(args.input)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = method.enhance([img_tensor])

    output = enhanced.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, output_bgr)
    print(f"Enhanced image saved to {args.output}")


if __name__ == "__main__":
    main()

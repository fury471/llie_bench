import torch
import cv2
import numpy as np
import os
import argparse
from core.registry import METHOD_REGISTRY, lookup

# import plugins so they register themselves
import plugins

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--method", type=str, required=True, help="Method name e.g. zerodce, clahe")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (optional)")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="results/enhanced.png", help="Path to save output")
    args = parser.parse_args()

    # load method from registry
    method = lookup(METHOD_REGISTRY, args.method)()
    if args.ckpt:
        method.load_ckpt(args.ckpt)
    method.eval()

    # load and preprocess image
    img = cv2.imread(args.input)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    # enhance
    with torch.no_grad():
        enhanced = method.enhance([img_tensor])

    # save output
    output = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, output_bgr)
    print(f"Enhanced image saved to {args.output}")

if __name__ == "__main__":
    main()
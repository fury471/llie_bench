import torch
import cv2
import numpy as np
import os
from methods.zerodce import ZeroDCE

def main():
    # 1. Initialize model and load your saved weights
    model = ZeroDCE()
    model.load_ckpt("checkpoints/zerodce_lolv1/checkpoint_epoch_012.pth")
    model.eval()

    # 2. Load and preprocess a dark image
    # Replace this path with whatever image you want to enhance
    img_path = "data/LOLdataset/eval15/low/1.png" 
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to PyTorch Tensor (1, 3, H, W)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    # 3. Produce the enhanced output
    with torch.no_grad():
        # The model's forward pass expects a list where batch[0] is the input
        curve_params = model([img_tensor])

        # 4. Apply the Zero-DCE iterative curve mapping
        # We split the 24 channels into 8 steps of 3 channels (RGB)
        enhanced_tensor = img_tensor.clone()
        for i in range(8):
            A = curve_params[:, i*3:(i+1)*3, :, :]
            # The Zero-DCE enhancement formula: LE(x) = x + A * x * (1 - x)
            enhanced_tensor = enhanced_tensor + A * enhanced_tensor * (1.0 - enhanced_tensor)

    # 5. Post-process to save as an image file
    output_numpy = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_numpy = np.clip(output_numpy * 255.0, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR)

    # Ensure the results folder exists (as listed in your .gitignore)
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/enhanced_photo.png", output_bgr)
    print("Success! Enhanced image saved to results/enhanced_photo.png")

if __name__ == "__main__":
    main()
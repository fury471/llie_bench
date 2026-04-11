import torch
import plugins
from core.registry import METHOD_REGISTRY, lookup
import cv2, numpy as np
import torch.nn.functional as F

model = lookup(METHOD_REGISTRY, "retinexnet")()
model.load_ckpt("checkpoints/retinexnet_lolv1_relight/checkpoint_epoch_006.pth")
model.eval()

img = cv2.imread("data/LOLdataset/eval15/low/493.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    R, I = model.decom_net(img_tensor)
    print(f"R: R={R[0,0].mean():.3f}, G={R[0,1].mean():.3f}, B={R[0,2].mean():.3f}")
    print(f"I mean: {I.mean():.3f}")
    I_delta = model.relight_net(I, R)
    if I_delta.shape != I.shape:
        I_delta = F.interpolate(I_delta, size=I.shape[2:], mode='bilinear', align_corners=False)
    print(f"I_delta: mean={I_delta.mean():.3f}, min={I_delta.min():.3f}, max={I_delta.max():.3f}")
    enhanced = R * I_delta
    print(f"enhanced: R={enhanced[0,0].mean():.3f}, G={enhanced[0,1].mean():.3f}, B={enhanced[0,2].mean():.3f}")
import gradio as gr
import torch
import numpy as np
import plugins
from core.registry import METHOD_REGISTRY, lookup

def enhance(image, method_name, ckpt_dir):
    # load method from registry
    method = lookup(METHOD_REGISTRY, method_name)()
    
    # auto-load latest checkpoint if available
    from pathlib import Path
    if ckpt_dir:
        ckpts = sorted(Path(ckpt_dir).glob("*.pth"))
        if ckpts:
            method.load_ckpt(str(ckpts[-1]))
    
    method.eval()

    # convert numpy image to tensor
    img = image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    # enhance
    with torch.no_grad():
        enhanced = method.enhance([img_tensor])

    # convert back to numpy
    output = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return output

demo = gr.Interface(
    fn=enhance,
    inputs=[
        gr.Image(type="numpy", label="Input Dark Image"),
        gr.Dropdown(choices=["zerodce", "clahe"], value="zerodce", label="Method"),
        gr.Textbox(value="checkpoints/zerodce_lolv1", label="Checkpoint Directory (optional)")
    ],
    outputs=gr.Image(type="numpy", label="Enhanced Image"),
    title="LLIE Bench Demo",
    description="Upload a low-light image and select a method to enhance it."
)

if __name__ == "__main__":
    demo.launch()
"""Gradio demo for the version 2 U-Net (32 base channels, 128 by 128 input, trained on the lesion split, seed 42)."""

import gradio as gr
import numpy as np
import torch
from PIL import Image

from unet import UNet

SIZE = 128
# Weights from the GitHub repo, pinned to the version 2 merge commit
WEIGHTS = "https://raw.githubusercontent.com/Boatengs/skin-lesion-segmentation/2e96cf6/models/unet32_lesion_split.pth"
model = UNet(base=32)
model.load_state_dict(torch.hub.load_state_dict_from_url(WEIGHTS, map_location="cpu", file_name="unet32_lesion_split.pth"))
model.eval()


def segment(image):
    if image is None:
        return None, ""
    im = Image.fromarray(image).convert("RGB")
    x = torch.from_numpy(np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR))).permute(2, 0, 1).float()[None] / 255.0
    with torch.inference_mode():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
    mask = Image.fromarray((prob > 0.5).astype(np.uint8) * 255).resize(im.size, Image.BILINEAR)
    m = np.asarray(mask) > 127
    out = np.asarray(im).copy()
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 64, 160])).astype(np.uint8)
    return out, f"Lesion covers {100 * m.mean():.1f}% of the photo."


demo = gr.Interface(
    segment,
    gr.Image(label="Dermoscopy photo"),
    [gr.Image(label="Predicted lesion outline"), gr.Textbox(label="Area")],
    title="Skin lesion segmentation",
    description=(
        "A U-Net trained on HAM10000 with every photo of a lesion kept on one side of the split. "
        "Mean test Dice 0.936 over three seeds on lesions it never saw. It outlines lesions; it does not diagnose them. "
        "Details: https://github.com/Boatengs/skin-lesion-segmentation"
    ),
)

if __name__ == "__main__":
    demo.launch()

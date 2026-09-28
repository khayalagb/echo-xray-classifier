"""Grad-CAM on a sample of correct + misclassified test images (README #3).

ponytail: pytorch-grad-cam already does this well; wrapping it in a custom
implementation would just be a worse copy of the library.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent.parent


def save_gradcam_samples(model, df, transform, out_dir: Path, n=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    for _, row in df.head(n).iterrows():
        img = Image.open(ROOT / "data" / "processed" / row["path"]).convert("L")
        x = transform(img).unsqueeze(0)
        grayscale_cam = cam(input_tensor=x)[0]
        rgb = np.repeat(np.asarray(img.resize((224, 224)))[..., None], 3, axis=2).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        Image.fromarray(overlay).save(out_dir / f"gradcam_{Path(row['path']).stem}.png")

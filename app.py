"""Public demo: upload an echo frame or chest X-ray, get a prediction + Grad-CAM.

ponytail: one file, no framework beyond Streamlit. Models aren't committed to
git (keeps the repo small, same call as the dataset) -- this downloads the
trained checkpoint from the GitHub Release on first run and caches it for the
life of the container.
"""
import json
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import CLASSES
from src.model import build_model
from src.transforms import build_transforms

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://github.com/khayalagb/echo-xray-classifier/releases/download/models-v1/model.pt"
OOD_THRESHOLD = 0.6  # same cutoff as src/evaluate.py


@st.cache_resource
def load_model():
    model_path = ROOT / "outputs" / "model.pt"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    meta = json.load(open(ROOT / "outputs" / "train_meta.json"))
    return model, meta


def predict(model, meta, img: Image.Image):
    transform = build_transforms(meta["mean"], meta["std"], train=False)
    gray = img.convert("L")
    x = transform(gray).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    pred_idx = int(probs.argmax())

    # Grad-CAM against the *predicted* class -- at inference time we don't have
    # a true label to target, unlike src/gradcam.py's offline eval usage
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb = np.repeat(np.asarray(gray.resize((224, 224)))[..., None], 3, axis=2).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    return pred_idx, probs, overlay


st.set_page_config(page_title="Echo / X-ray classifier", page_icon="\U0001fac0")
st.title("Echo / X-ray classifier")
st.caption(
    "ResNet18 classifying apical 2-chamber echo (A2C), apical 4-chamber echo (A4C), "
    "or chest X-ray. See the "
    "[repo](https://github.com/khayalagb/echo-xray-classifier) for training/eval details."
)

model, meta = load_model()
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file is not None:
    img = Image.open(file)
    pred_idx, probs, overlay = predict(model, meta, img)
    max_prob = float(probs[pred_idx])

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM (evidence for predicted class)", use_container_width=True)

    if max_prob < OOD_THRESHOLD:
        st.warning(f"Low confidence ({max_prob:.2f}) — likely not one of the three trained classes (OOD).")
    else:
        st.success(f"Prediction: **{CLASSES[pred_idx]}** ({max_prob:.2%} confidence)")

    st.bar_chart({cls: float(p) for cls, p in zip(CLASSES, probs)})

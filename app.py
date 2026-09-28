"""Public demo: upload echo frames / chest X-rays, get predictions + Grad-CAM.

ponytail: one file, no framework beyond Streamlit. Models aren't committed to
git (keeps the repo small, same call as the dataset) -- this downloads the
trained checkpoint from the GitHub Release on first run and caches it for the
life of the container. Batch upload just loops the single-image path per file
-- no need for a separate code path.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
ACCENT = "#2563EB"  # predicted class
MUTED = "#9CA3AF"   # everything else


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


def prob_barh(probs, pred_idx):
    """Horizontal bar plot of class probabilities, predicted class highlighted."""
    colors = [ACCENT if i == pred_idx else MUTED for i in range(len(CLASSES))]
    fig, ax = plt.subplots(figsize=(4, 1.6))
    y = np.arange(len(CLASSES))
    ax.barh(y, probs, color=colors, height=0.6)
    for i, p in enumerate(probs):
        ax.text(p + 0.02, i, f"{p:.0%}", va="center", fontsize=9, color="#374151")
    ax.set_yticks(y, CLASSES)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()  # CLASSES order top-to-bottom
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig


def render_result(name, img, model, meta):
    pred_idx, probs, overlay = predict(model, meta, img)
    max_prob = float(probs[pred_idx])

    col1, col2, col3 = st.columns([1, 1, 1.2])
    with col1:
        st.image(img, caption="Input", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM", use_container_width=True)
    with col3:
        if max_prob < OOD_THRESHOLD:
            st.warning(f"Low confidence ({max_prob:.2f}) — likely not one of the trained classes (OOD).")
        else:
            st.success(f"**{CLASSES[pred_idx]}** ({max_prob:.2%} confidence)")
        st.pyplot(prob_barh(probs, pred_idx), use_container_width=True)


st.set_page_config(page_title="Echo / X-ray classifier", page_icon="\U0001fac0", layout="wide")
st.title("Echo / X-ray classifier")
st.caption(
    "ResNet18 classifying apical 2-chamber echo (A2C), apical 4-chamber echo (A4C), "
    "or chest X-ray. See the "
    "[repo](https://github.com/khayalagb/echo-xray-classifier) for training/eval details."
)

model, meta = load_model()
files = st.file_uploader(
    "Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if files:
    for file in files:
        with st.expander(file.name, expanded=len(files) <= 5):
            render_result(file.name, Image.open(file), model, meta)

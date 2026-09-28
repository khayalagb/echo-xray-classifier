"""Test-set evaluation: metrics, bootstrap CI, OOD threshold, Grad-CAM samples.

ponytail: bootstrap CI is 200 lines in a paper, 8 lines with np.random.choice.
No stats library needed for a percentile bootstrap.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import CLASSES, EchoXrayDataset, load_manifest, patient_level_split
from src.gradcam import save_gradcam_samples
from src.model import build_model
from src.train import DEVICE
from src.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent
OOD_THRESHOLD = 0.6  # softmax max-prob below this -> flag as "unknown" (README #4)


def bootstrap_ci(y_true, y_pred, label_idx, n=2000, seed=42):
    """Percentile bootstrap 95% CI for per-class F1."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores.append(f1_score(y_true[idx] == label_idx, y_pred[idx] == label_idx, zero_division=0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    meta = json.load(open(ROOT / "outputs" / "train_meta.json"))
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ROOT / "outputs" / "model.pt", map_location=DEVICE))
    model.eval()

    df = load_manifest()
    test_df = patient_level_split(df)["test"]
    transform = build_transforms(meta["mean"], meta["std"], train=False)
    loader = DataLoader(EchoXrayDataset(test_df, transform), batch_size=32)

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(DEVICE)), dim=1).cpu().numpy()
            probs.append(out)
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = probs.argmax(1)
    max_prob = probs.max(1)
    ood_flags = max_prob < OOD_THRESHOLD

    results = {
        "accuracy": float((preds == labels).mean()),
        "roc_auc_ovr": float(roc_auc_score(labels, probs, multi_class="ovr")),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "ood_flagged_count": int(ood_flags.sum()),
        "per_class": {},
    }
    for i, cls in enumerate(CLASSES):
        f1 = f1_score(labels == i, preds == i, zero_division=0)
        lo, hi = bootstrap_ci(labels, preds, i)
        results["per_class"][cls] = {"f1": float(f1), "f1_ci95": [lo, hi]}

    print(json.dumps(results, indent=2))
    json.dump(results, open(ROOT / "outputs" / "test_results.json", "w"), indent=2)

    misclassified = test_df[preds != labels]
    correct = test_df[preds == labels]
    save_gradcam_samples(model, misclassified, transform, ROOT / "outputs" / "gradcam" / "misclassified")
    save_gradcam_samples(model, correct, transform, ROOT / "outputs" / "gradcam" / "correct")


if __name__ == "__main__":
    main()

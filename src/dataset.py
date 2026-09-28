"""Patient-level split + PyTorch Dataset for the processed manifest.

ponytail: a CAMUS patient contributes exactly one A2C row and one A4C row
(same patient_id, two labels — that's correct, it's the same heart from two
probe angles), so "one patient_id -> one label" doesn't hold for CAMUS.
What must hold, and does, is grouping: split on patient_id, not on label, so
a patient's A2C and A4C never land in different splits. Stratify by *source*
(camus / nih / covid) instead of by label — since each source contributes a
fixed, uniform label pattern per patient, splitting sources proportionally
already splits labels proportionally.
"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
CLASSES = ["A2C", "A4C", "XRAY"]


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "manifest.csv")


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient_id, tagged with its source (for stratification)."""
    patients = df.drop_duplicates("patient_id")[["patient_id"]].copy()
    patients["source"] = patients["patient_id"].str.split("_").str[0]
    return patients


def patient_level_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """70/15/15 split, grouped by patient_id, stratified by source."""
    patients = _patient_table(df)
    train_p, rest_p = train_test_split(
        patients, test_size=val_frac + test_frac, stratify=patients["source"], random_state=seed
    )
    val_p, test_p = train_test_split(
        rest_p, test_size=test_frac / (val_frac + test_frac), stratify=rest_p["source"], random_state=seed
    )
    splits = {}
    for name, pids in (("train", train_p), ("val", val_p), ("test", test_p)):
        splits[name] = df[df["patient_id"].isin(pids["patient_id"])].reset_index(drop=True)
    return splits


def kfold_patient_splits(train_df: pd.DataFrame, k=5, seed=42):
    """Stratified k-fold over the *training* patients (grouped, source-stratified)."""
    patients = _patient_table(train_df).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(patients["patient_id"], patients["source"]):
        tr_pids = set(patients.loc[tr_idx, "patient_id"])
        va_pids = set(patients.loc[va_idx, "patient_id"])
        yield (
            train_df[train_df["patient_id"].isin(tr_pids)].reset_index(drop=True),
            train_df[train_df["patient_id"].isin(va_pids)].reset_index(drop=True),
        )


def compute_mean_std(df: pd.DataFrame) -> tuple[float, float]:
    """Dataset-computed grayscale mean/std (see README #7 — not ImageNet stats)."""
    import numpy as np

    sample = df["path"].sample(min(300, len(df)), random_state=42)
    means, stds = [], []
    for p in sample:
        arr = np.asarray(Image.open(PROCESSED / p).convert("L"), dtype=np.float32) / 255.0
        means.append(arr.mean())
        stds.append(arr.std())
    return float(np.mean(means)), float(np.mean(stds))


class EchoXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(PROCESSED / row["path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        label = CLASSES.index(row["label"])
        return img, torch.tensor(label, dtype=torch.long)

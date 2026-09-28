"""Raw data -> PNG + manifest.csv.

ponytail: one function per real source format, no shared "loader" abstraction
for three formats that don't actually share any loading logic.

Source formats actually downloaded (differ from the CREATIS raw NIfTI layout
because CAMUS was pulled from a Kaggle mirror instead, see README):
  - CAMUS: single HDF5 file, `train {2ch,4ch} {frames,masks}`, 450 patients x
    (ED, ES) pairs, frames pre-normalized to roughly zero mean / unit std.
    No patient IDs and no ED/ES flag are stored — both are recovered below.
  - NIH ChestX-ray14: standard layout, `images_NNN/images/*.png` +
    `Data_Entry_2017.csv` (has real Patient ID column).
  - COVID-19 chest X-ray repo: flat folder of images, no patient metadata,
    treated as one image = one unit (see README caveat).
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def _to_png(arr: np.ndarray, dst: Path) -> None:
    """Percentile-stretch to 0-255 uint8 (handles the CAMUS mirror's z-score floats
    and raw uint8/uint16 X-rays the same way)."""
    arr = arr.astype(np.float32).squeeze()
    lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    Image.fromarray(arr.astype(np.uint8)).save(dst)


def convert_camus(rows: list) -> None:
    """Pick the ED frame of each (ED, ES) pair using LV-cavity area (ED = larger
    cavity, end of diastole = fully filled). Verified 900/900 on this file: the
    even index of every pair has strictly larger label-1 area than the odd one
    — see the session notes / commit message for the check that established this,
    since the mirror itself ships no ED/ES label."""
    path = RAW / "camus" / "image_dataset.hdf5"
    with h5py.File(path, "r") as f:
        for view, label in (("2ch", "A2C"), ("4ch", "A4C")):
            frames = f[f"train {view} frames"]
            masks = f[f"train {view} masks"]
            n_patients = frames.shape[0] // 2
            dst_dir = OUT / "images" / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in range(n_patients):
                area0 = int(np.sum(masks[2 * p] == 1))
                area1 = int(np.sum(masks[2 * p + 1] == 1))
                ed_idx = 2 * p if area0 >= area1 else 2 * p + 1
                pid = f"camus_{p:03d}"
                dst = dst_dir / f"{pid}_{label}.png"
                _to_png(np.asarray(frames[ed_idx]), dst)
                rows.append({"path": str(dst.relative_to(OUT)), "label": label, "patient_id": pid})


def convert_nih_xray(rows: list, n: int, seed: int = 42) -> None:
    entry = pd.read_csv(RAW / "nih_cxr" / "Data_Entry_2017.csv")
    id_by_file = dict(zip(entry["Image Index"], entry["Patient ID"]))
    files = sorted((RAW / "nih_cxr").glob("images_*/images/*.png"))
    random.Random(seed).shuffle(files)
    dst_dir = OUT / "images" / "XRAY"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files[:n]:
        pid = id_by_file.get(f.name)
        if pid is None:
            continue
        dst = dst_dir / f"nih_{f.stem}.png"
        if not dst.exists():  # resumable: this step is slow on a large mounted folder
            Image.open(f).convert("L").save(dst)
        rows.append({"path": str(dst.relative_to(OUT)), "label": "XRAY", "patient_id": f"nih_{pid}"})


def convert_covid_xray(rows: list, n: int, seed: int = 42) -> None:
    """No patient metadata ships with this repo copy, so one image = one group
    (documented limitation, see README — this is the one place the split isn't
    truly patient-level)."""
    files = sorted(p for p in (RAW / "covid_cxr").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    random.Random(seed).shuffle(files)
    dst_dir = OUT / "images" / "XRAY"
    for f in files[:n]:
        dst = dst_dir / f"covid_{f.stem}.png"
        if not dst.exists():
            Image.open(f).convert("L").save(dst)
        rows.append({"path": str(dst.relative_to(OUT)), "label": "XRAY", "patient_id": f"covid_{f.stem}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nih-count", type=int, default=1000)
    parser.add_argument("--covid-count", type=int, default=200)
    parser.add_argument("--fresh", action="store_true", help="wipe data/processed/images first instead of resuming")
    args = parser.parse_args()

    if args.fresh and (OUT / "images").exists():
        shutil.rmtree(OUT / "images")
    rows = []
    convert_camus(rows)
    convert_nih_xray(rows, args.nih_count)
    convert_covid_xray(rows, args.covid_count)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "patient_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images -> {OUT / 'manifest.csv'}")
    for label in ("A2C", "A4C", "XRAY"):
        print(f"  {label}: {sum(1 for r in rows if r['label'] == label)}")


if __name__ == "__main__":
    main()

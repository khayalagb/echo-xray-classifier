# Echo View Classifier (A2C / A4C / Chest X-ray)

3-class image classifier: apical 2-chamber echo (A2C), apical 4-chamber echo (A4C), chest X-ray.
ResNet18, fine-tuned, Optuna-tuned, exported to ONNX + quantized.

Rebuilt version of a technical-assessment project, with fixes over the original submission.
Each fix below exists because of a specific weakness in v1 — not novelty for its own sake.

## Data

- Echo: CAMUS dataset (ED frames only, A2C + A4C), https://humanheart-project.creatis.insa-lyon.fr/database
  Pulled from a Kaggle mirror (the CREATIS site itself is not reachable from this network) as a single
  `image_dataset.hdf5` containing only the official CAMUS **train** split — 450 patients per view, 900
  frames per view (ED+ES pairs). The official 500-patient set reserves 50 patients as a held-out test set
  that this mirror doesn't ship, so 450/450 is what's available here, not a subsampling choice.
  The file has no patient IDs or ED/ES labels either: patient_id is reconstructed as the pair index
  (frames are stored as consecutive (ED, ES) pairs), and ED vs ES is recovered from the segmentation
  masks — ED is the frame with the larger left-ventricle cavity area, verified 900/900 against the mask data.
  Requires a free account + data use agreement — no programmatic download, see `scripts/download_data.md`.
- Chest X-ray: NIH ChestX-ray14 subset + COVID-19 Chest X-Ray Image Repository.

Data is never committed (`data/` is gitignored). `scripts/download_data.md` has exact steps.

## What changed vs v1, and why

| # | Change | Why |
|---|---|---|
| 1 | Patient-level split (`GroupShuffleSplit` on CAMUS patient ID) | v1 split by image, not patient. CAMUS has 1 ED frame/patient/view so leakage risk was low in practice, but patient-level is the correct default for any medical-imaging split — a model that's never right by accident. |
| 2 | Ultrasound-appropriate augmentation: gamma jitter, speckle (multiplicative) noise, small rotation/affine. Dropped hue/saturation jitter. | Hue/saturation are near no-ops on grayscale-replicated images — pure waste. Gamma and speckle noise actually resemble real scanner/gain variation. |
| 3 | Grad-CAM on correct + misclassified samples | Confirms the model looks at cardiac structure, not borders/overlays/artifacts. Standard sanity check for medical imaging models. |
| 4 | OOD flag via softmax max-prob / entropy threshold | v1 forces every input into one of 3 classes, including images that are none of the three. Not acceptable for anything production-adjacent. |
| 5 | Stratified k-fold CV on the training split (in addition to the held-out test set) | 500 images/class is small; a single split has real split-luck variance. CV gives a more honest estimate before touching the test set once at the end. |
| 6 | Static (calibration-based) quantization compared against dynamic | Dynamic only compresses weights; static also quantizes activations and is usually faster for CNNs at similar accuracy. Both are benchmarked, not assumed. |
| 7 | Normalization is dataset-computed mean/std, documented explicitly | v1 used ImageNet mean/std without checking whether that's appropriate for grayscale medical images; dataset stats are more principled here. |
| 8 | Bootstrap 95% CI on per-class metrics | 75 test images/class means a point estimate like "F1 = 0.973" hides real uncertainty. CI makes that honest. |

Kept from v1 (already correct): ResNet18 ImageNet-pretrained backbone, cross-entropy loss, Optuna
hyperparameter search, F1 as the tuning objective (class imbalance), ONNX export, saving
misclassified images for error analysis.

## Layout

```
src/
  dataset.py      CAMUS + X-ray loading, patient-level split, k-fold CV splits
  transforms.py   grayscale-ultrasound augmentation pipeline
  model.py        ResNet18 factory
  train.py        Optuna sweep + CV training loop
  evaluate.py      test-set metrics, bootstrap CI, OOD threshold
  gradcam.py      Grad-CAM visualization
  export.py       ONNX export, dynamic vs static quantization benchmark
scripts/
  download_data.md   manual data-acquisition steps (no API for CAMUS)
  build_dataset.py   NIfTI -> PNG conversion, folder layout, manifest CSV
tests/            one smoke test per module (ponytail: no test for trivial code)
```

## Pipeline (run in order)

1. Follow `scripts/download_data.md` to get raw CAMUS + X-ray data under `data/raw/`.
2. `python scripts/build_dataset.py` — extracts ED frames from CAMUS NIfTI volumes, converts
   everything to PNG, writes `data/processed/manifest.csv` (path, label, patient_id, split).
3. `python -m src.train` — Optuna sweep, k-fold CV, trains final model on full train split.
4. `python -m src.evaluate` — test metrics + bootstrap CI + OOD threshold + Grad-CAM samples.
5. `python -m src.export` — ONNX export, dynamic vs static quantization benchmark.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

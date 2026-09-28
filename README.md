# Echo View Classifier (A2C / A4C / Chest X-ray)

A 3-class image classifier that tells apart two echocardiogram views — apical 2-chamber (A2C),
apical 4-chamber (A4C) — from chest X-rays. ResNet18 backbone, fine-tuned and hyperparameter-tuned
with Optuna + cross-validation, exported to ONNX and quantized for deployment.

## Data

- Echo: CAMUS dataset (ED frames only, A2C + A4C), https://humanheart-project.creatis.insa-lyon.fr/database
  Pulled from a Kaggle mirror (the CREATIS site itself was not reachable from this network) as a single
  `image_dataset.hdf5` containing only the official CAMUS **train** split — 450 patients per view, 900
  frames per view (ED+ES pairs). The official 500-patient set reserves 50 patients as a held-out test set
  that this mirror doesn't ship, so 450/450 is what's available here.
  The file ships no patient IDs or ED/ES labels: patient_id is reconstructed from the pair index
  (frames are stored as consecutive (ED, ES) pairs), and ED vs ES is recovered from the segmentation
  masks — ED is the frame with the larger left-ventricle cavity area, verified 900/900 against the mask data.
  Requires a free account + data use agreement — no programmatic download, see `scripts/download_data.md`.
- Chest X-ray: NIH ChestX-ray14 subset + COVID-19 Chest X-Ray Image Repository.

Data is never committed (`data/` is gitignored). `scripts/download_data.md` has exact steps.

## Design decisions

Medical imaging has a few failure modes that are easy to miss and expensive to get wrong, so
each choice below is deliberate rather than default:

| # | Decision | Why |
|---|---|---|
| 1 | Patient-level split, grouped by `patient_id`, stratified by data source | A model should never see the same patient in both train and test — even when the leakage risk looks small (CAMUS has one ED frame per patient per view), patient-level splitting is the correct default for any medical-imaging dataset. |
| 2 | Ultrasound-appropriate augmentation: gamma jitter, speckle (multiplicative) noise, small rotation/affine. No hue/saturation jitter. | Hue/saturation are near no-ops on grayscale-replicated images. Gamma and speckle noise actually resemble real scanner gain and acquisition noise. |
| 3 | Grad-CAM on correct + misclassified samples | Confirms the model is attending to cardiac structure, not borders, overlays, or scan artifacts — a standard sanity check for medical imaging models. |
| 4 | OOD flag via softmax max-probability threshold | A classifier that forces every input into one of three classes will confidently mislabel anything it's never seen. Flagging low-confidence predictions as "unknown" is essential before anything production-adjacent. |
| 5 | Stratified k-fold CV on the training split, in addition to the held-out test set | With only ~450 images per echo class, a single validation split carries real split-luck variance. CV gives Optuna a more honest signal before the test set is touched once, at the end. |
| 6 | Static (calibration-based) quantization | Quantizes both weights and activations after a calibration pass, which is typically the faster option for CNN inference — the standard choice for this kind of model. |
| 7 | Normalization uses dataset-computed mean/std, not ImageNet stats | Dataset-specific statistics are more principled than reusing ImageNet's for grayscale medical images. |
| 8 | Bootstrap 95% CI on per-class metrics | With ~70 test images per class, a point estimate like "F1 = 0.97" hides real uncertainty. A confidence interval makes that honest. |

Also used: ResNet18 (ImageNet-pretrained), cross-entropy loss, Optuna hyperparameter search with
F1 as the tuning objective (class imbalance), ONNX export, and saved misclassified images for
error analysis.

## Layout

```
src/
  dataset.py      CAMUS + X-ray loading, patient-level split, k-fold CV splits
  transforms.py   grayscale-ultrasound augmentation pipeline
  model.py        ResNet18 factory
  train.py        Optuna sweep + CV training loop
  evaluate.py     test-set metrics, bootstrap CI, OOD threshold, triggers Grad-CAM
  gradcam.py      Grad-CAM visualization
  export.py       ONNX export + static quantization benchmark
scripts/
  download_data.md   manual data-acquisition steps (no API for CAMUS)
  build_dataset.py   raw data -> PNG conversion, manifest CSV
tests/            one smoke test per module
```

## Pipeline (run in order)

1. Follow `scripts/download_data.md` to get raw CAMUS + X-ray data under `data/raw/`.
2. `python scripts/build_dataset.py` — converts everything to PNG, writes
   `data/processed/manifest.csv` (path, label, patient_id).
3. `python -m src.train` — Optuna sweep with k-fold CV, then trains a final model on the full train split.
4. `python -m src.evaluate` — test metrics + bootstrap CI + OOD threshold + Grad-CAM samples.
5. `python -m src.export` — ONNX export + static quantization benchmark.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

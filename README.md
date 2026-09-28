# Medical Image Classifier (A2C / A4C / Chest X-ray)

A 3-class image classifier that tells apart two echocardiogram views — apical 2-chamber (A2C),
apical 4-chamber (A4C) — from chest X-rays. ResNet18 backbone, fine-tuned and hyperparameter-tuned
with Optuna + cross-validation, exported to ONNX and quantized for deployment.

## Data

- Echo: CAMUS dataset (ED frames only, A2C + A4C). Pulled from a Kaggle [mirror](https://www.kaggle.com/datasets/shoybhasan/camus-human-heart-data/data) containing only the official CAMUS **train** split — 450 patients per view, 900 frames per view (ED+ES pairs).
- Chest X-ray: [NIH ChestX-ray14 subset](https://www.kaggle.com/datasets/nih-chest-xrays/data/data) + [COVID-19 Chest X-Ray Image Repository](https://github.com/ieee8023/covid-chestxray-dataset).

Prebuilt processed dataset (2100 PNGs + `manifest.csv`, 655MB): [download](https://github.com/khayalagb/echo-xray-classifier/releases/download/data-v1/processed.zip) ([release page](https://github.com/khayalagb/echo-xray-classifier/releases/tag/data-v1)). Unzip into `data/processed/` to skip `scripts/build_dataset.py` entirely.

## Pretrained models

Skip training entirely and grab the already-trained weights from the [models-v1 release](https://github.com/khayalagb/echo-xray-classifier/releases/tag/models-v1):

- [`model.pt`](https://github.com/khayalagb/echo-xray-classifier/releases/download/models-v1/model.pt) (43MB) — fp32 checkpoint, `state_dict` for `src.model.build_model()`.
- [`model.onnx`](https://github.com/khayalagb/echo-xray-classifier/releases/download/models-v1/model.onnx) (43MB) — ONNX export, runnable without PyTorch via `onnxruntime`.
- [`model_static_quant.pt`](https://github.com/khayalagb/echo-xray-classifier/releases/download/models-v1/model_static_quant.pt) (11MB) — int8 statically quantized (qnnpack), ~15% faster inference; no autograd support, so it can't drive Grad-CAM.

Benchmark (`outputs/quantization_benchmark.json`, ms/inference on Apple Silicon CPU): fp32 PyTorch 10.7, ONNX 12.2, static quantized 9.1.

## Demo

A Streamlit app (`app.py`) for interactive use: upload one or more images (batch supported), see the prediction, an OOD flag if confidence is low, the Grad-CAM overlay, and a horizontal bar chart of class probabilities per image. It downloads `model.pt` from the [models-v1 release](https://github.com/khayalagb/echo-xray-classifier/releases/tag/models-v1) on first run rather than needing local training first.

```
pip install -r requirements-app.txt
streamlit run app.py
```

(`requirements-app.txt` is the slim subset `app.py` actually imports — training/eval/export need
the full `requirements.txt`.)

## Design decisions

Medical imaging has a few failure modes that are easy to miss and expensive to get wrong, so
each choice below is deliberate rather than default:

| # | Decision | Why |
|---|---|---|
| 1 | Patient-level split, grouped by `patient_id`, stratified by data source | A model should never see the same patient in both train and test — even when the leakage risk looks small (CAMUS has one ED frame per patient per view), patient-level splitting is the correct default for any medical-imaging dataset. |
| 2 | Ultrasound-appropriate augmentation: gamma jitter, speckle (multiplicative) noise, small rotation/affine. | Gamma and speckle noise actually resemble real scanner gain and acquisition noise. |
| 3 | Grad-CAM on correct + misclassified samples | Confirms the model is attending to cardiac structure, not borders, overlays, or scan artifacts — a standard sanity check for medical imaging models. |
| 4 | OOD flag via softmax max-probability threshold | A classifier that forces every input into one of three classes will confidently mislabel anything it's never seen. Flagging low-confidence predictions as "unknown" is essential. |
| 5 | Stratified k-fold CV on the training split, in addition to the held-out test set | With only ~450 images per echo class, a single validation split carries real split-luck variance. CV gives Optuna a more honest signal before the test set is touched once, at the end. |
| 6 | Static (calibration-based) quantization | Quantizes both weights and activations after a calibration pass, which is typically the faster option for CNN inference — the standard choice for this kind of model. |
| 7 | Normalization uses dataset-computed mean/std, not ImageNet stats | Dataset-specific statistics are more principled than reusing ImageNet's for grayscale medical images. |
| 8 | Bootstrap 95% CI on per-class metrics | With ~70 test images per class, a point estimate like "F1 = 0.97" hides real uncertainty. A confidence interval makes that honest. |

Also used: ResNet18 (ImageNet-pretrained), cross-entropy loss, Optuna hyperparameter search with
F1 as the tuning objective (class imbalance), ONNX export.

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
  build_dataset.py   raw data -> PNG conversion, manifest CSV
tests/            one smoke test per module
```

## Pipeline (run in order)

1. `python scripts/build_dataset.py` — converts everything to PNG, writes
   `data/processed/manifest.csv` (path, label, patient_id).
2. `python -m src.train` — Optuna sweep with k-fold CV, then trains a final model on the full train split.
3. `python -m src.evaluate` — test metrics + bootstrap CI + OOD threshold + Grad-CAM samples.
4. `python -m src.export` — ONNX export + static quantization benchmark.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

[MIT](LICENSE)

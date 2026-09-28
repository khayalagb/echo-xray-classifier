# Getting the raw data

Nothing here can be scripted end-to-end — both sources require accepting a data-use agreement
in a browser first. Manual steps below.

## CAMUS (echo, A2C + A4C)

1. Create a free account at https://humanheart-project.creatis.insa-lyon.fr/database
2. Open the CAMUS collection (the link your tech lead sent):
   https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8
3. Download the training archive. It unpacks to one folder per patient:
   `patientXXXX/patientXXXX_2CH_ED.nii.gz`, `patientXXXX_4CH_ED.nii.gz` (+ `_gt.nii.gz` masks, unused here).
4. Extract into `data/raw/camus/` so you end up with `data/raw/camus/patient0001/...`, etc.
   The patient folder name is the patient ID `build_dataset.py` uses for the group split — don't rename it.

## Chest X-ray

- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC (or the Kaggle mirror) — grab any
  1000 images, drop into `data/raw/nih_cxr/`.
- COVID-19 Chest X-Ray Image Repository: https://github.com/ieee8023/covid-chestxray-dataset —
  clone or download the `images/` folder, put ~200 images into `data/raw/covid_cxr/`.

No patient IDs are available for either X-ray source, so `build_dataset.py` uses a plain
stratified image-level split for the X-ray class only — CAMUS is the one that gets grouped by patient.

Once all three folders are populated, run `python scripts/build_dataset.py`.

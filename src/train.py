"""Optuna sweep with stratified-group k-fold CV, then a final fit on the full train split.

ponytail: one objective function, one study. CV score (mean F1 across folds) is what
Optuna optimizes, so the chosen hyperparameters aren't a fluke of one split (README #5).
Early stopping via Optuna's MedianPruner, same as v1.

Progress logging: a 50-trial x 5-fold sweep can run for hours with the original code
printing nothing in between trials. Every epoch now prints a timestamped one-liner and
overwrites outputs/progress.json, so progress is checkable from another shell/session
without needing to attach to whatever terminal launched it (`cat outputs/progress.json`
or `tail -f` the log file if you redirected stdout to one).
"""
import argparse
import json
import time
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import EchoXrayDataset, compute_mean_std, kfold_patient_splits, load_manifest, patient_level_split
from src.model import build_model
from src.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent
PROGRESS_PATH = ROOT / "outputs" / "progress.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_progress(**fields) -> None:
    PROGRESS_PATH.parent.mkdir(exist_ok=True)
    payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2))


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            all_preds += out.argmax(1).cpu().tolist()
            all_labels += y.cpu().tolist()
    from sklearn.metrics import f1_score

    return f1_score(all_labels, all_preds, average="macro")


def fit(train_df, val_df, mean, std, lr, batch_size, epochs, trial=None, tag="final"):
    train_ds = EchoXrayDataset(train_df, build_transforms(mean, std, train=True))
    val_ds = EchoXrayDataset(val_df, build_transforms(mean, std, train=False))
    import os
    workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers, persistent_workers=True)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        run_epoch(model, train_loader, optimizer)
        val_f1 = run_epoch(model, val_loader)
        best_f1 = max(best_f1, val_f1)
        dt = time.time() - t0
        _log(f"{tag} | epoch {epoch + 1}/{epochs} | val macro-F1={val_f1:.4f} | best={best_f1:.4f} | {dt:.1f}s/epoch")
        _write_progress(stage=tag, epoch=epoch + 1, epochs=epochs, val_f1=val_f1, best_f1=best_f1, seconds_per_epoch=round(dt, 1))
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                _log(f"{tag} | pruned")
                raise optuna.TrialPruned()
    return model, best_f1


def objective(trial, train_df, mean, std, k, n_trials):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
    epochs = trial.suggest_int("epochs", 5, 20)
    _log(f"trial {trial.number + 1}/{n_trials} | lr={lr:.2e} batch_size={batch_size} epochs={epochs}")

    fold_scores = []
    for fold_idx, (fold_train, fold_val) in enumerate(kfold_patient_splits(train_df, k=k)):
        tag = f"trial {trial.number + 1}/{n_trials} fold {fold_idx + 1}/{k}"
        _, f1 = fit(fold_train, fold_val, mean, std, lr, batch_size, epochs, trial=trial, tag=tag)
        fold_scores.append(f1)
        _log(f"trial {trial.number + 1}/{n_trials} fold {fold_idx + 1}/{k} done | fold F1={f1:.4f}")
    mean_f1 = sum(fold_scores) / len(fold_scores)
    _log(f"trial {trial.number + 1}/{n_trials} complete | mean CV F1={mean_f1:.4f}")
    return mean_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=3)  # 3 over 5: see README rationale / session notes
    args = parser.parse_args()

    df = load_manifest()
    splits = patient_level_split(df)
    mean, std = compute_mean_std(splits["train"])
    _log(f"dataset mean={mean:.4f} std={std:.4f} (used instead of ImageNet stats, see README #7)")
    _log(f"device={DEVICE} | train/val/test patients: "
         f"{splits['train']['patient_id'].nunique()}/{splits['val']['patient_id'].nunique()}/{splits['test']['patient_id'].nunique()}")
    _log(f"starting Optuna sweep: {args.trials} trials x {args.folds} folds")

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, splits["train"], mean, std, args.folds, args.trials), n_trials=args.trials)

    _log(f"sweep done | best params: {study.best_params}")
    best = study.best_params
    _log("training final model on full train split")
    final_model, val_f1 = fit(
        splits["train"], splits["val"], mean, std, best["lr"], best["batch_size"], best["epochs"], tag="final fit"
    )

    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    torch.save(final_model.state_dict(), ROOT / "outputs" / "model.pt")
    json.dump(
        {"best_params": best, "val_f1": val_f1, "mean": mean, "std": std},
        open(ROOT / "outputs" / "train_meta.json", "w"),
        indent=2,
    )
    _log(f"saved outputs/model.pt (val macro-F1={val_f1:.4f})")


if __name__ == "__main__":
    main()

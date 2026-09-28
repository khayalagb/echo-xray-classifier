"""Smoke tests. ponytail: assert-based, no fixtures/framework — this is a
sanity check on the split logic, not a full test suite."""
import pandas as pd

from src.dataset import patient_level_split


def test_patient_level_split_no_leakage_and_no_split_label():
    """A CAMUS-style patient contributes two rows (A2C + A4C, same patient_id) —
    make sure both always land in the same split, and no patient_id crosses splits."""
    df = pd.DataFrame(
        {
            "path": [f"img{i}.png" for i in range(40)],
            "label": ["A2C"] * 10 + ["A4C"] * 10 + ["XRAY"] * 20,
            "patient_id": [f"camus_{i}" for i in range(10)] * 2 + [f"xray_{i}" for i in range(20)],
        }
    )
    splits = patient_level_split(df, val_frac=0.2, test_frac=0.2, seed=0)
    train_p = set(splits["train"]["patient_id"])
    val_p = set(splits["val"]["patient_id"])
    test_p = set(splits["test"]["patient_id"])
    assert not (train_p & val_p) and not (train_p & test_p) and not (val_p & test_p)
    for name, split_df in splits.items():
        counts = split_df.groupby("patient_id")["label"].nunique()
        assert (counts <= 2).all()
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(df)


if __name__ == "__main__":
    test_patient_level_split_no_leakage_and_no_split_label()
    print("ok")

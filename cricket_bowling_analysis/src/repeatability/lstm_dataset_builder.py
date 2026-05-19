"""Build train/test LSTM datasets from delivery sequence NPY files."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _sequence_metadata_path(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.name.replace("_sequence.npy", "_metadata.csv"))


def _read_sequence_meta(npy_path: Path) -> dict:
    meta_path = _sequence_metadata_path(npy_path)
    if meta_path.exists():
        row = pd.read_csv(meta_path).iloc[0].to_dict()
        return {"bowler_id": str(row.get("bowler_id")), "delivery_id": str(row.get("delivery_id"))}
    stem = npy_path.stem.replace("_sequence", "")
    parts = stem.split("_", 1)
    return {"bowler_id": parts[0], "delivery_id": parts[1] if len(parts) > 1 else stem}


def build_lstm_dataset(
    sequence_npy_paths: Iterable[str],
    labels_csv_path,
    output_dir,
    split_mode: str = "by_bowler",
):
    """Create X/y train/test arrays and metadata CSVs."""
    labels = pd.read_csv(labels_csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_meta = [], [], []
    test_x, test_y, test_meta = [], [], []

    label_key = {
        (str(row["bowler_id"]), str(row["delivery_id"])): row
        for _, row in labels.iterrows()
    }

    for path in sequence_npy_paths:
        npy_path = Path(path)
        if not npy_path.exists():
            print(f"[REPEATABILITY] Warning: sequence not found: {npy_path}")
            continue
        meta = _read_sequence_meta(npy_path)
        key = (meta["bowler_id"], meta["delivery_id"])
        if key not in label_key:
            print(f"[REPEATABILITY] Warning: no label for {key}; skipping.")
            continue
        row = label_key[key]
        split = str(row.get("split", "train")).lower()
        label = int(row["label"])
        sequence = np.load(npy_path).astype(np.float32)
        record = {"bowler_id": key[0], "delivery_id": key[1], "path": str(npy_path), "label": label}
        if split == "test":
            test_x.append(sequence); test_y.append(label); test_meta.append(record)
        else:
            train_x.append(sequence); train_y.append(label); train_meta.append(record)

    train_bowlers = {m["bowler_id"] for m in train_meta}
    test_bowlers = {m["bowler_id"] for m in test_meta}
    overlap = train_bowlers.intersection(test_bowlers)
    if split_mode == "by_bowler" and overlap:
        print(f"[REPEATABILITY] Warning: bowler leakage across train/test: {sorted(overlap)}")

    np.save(output / "X_train.npy", np.asarray(train_x, dtype=np.float32))
    np.save(output / "y_train.npy", np.asarray(train_y, dtype=np.float32))
    np.save(output / "X_test.npy", np.asarray(test_x, dtype=np.float32))
    np.save(output / "y_test.npy", np.asarray(test_y, dtype=np.float32))
    pd.DataFrame(train_meta).to_csv(output / "metadata_train.csv", index=False)
    pd.DataFrame(test_meta).to_csv(output / "metadata_test.csv", index=False)
    print(f"[REPEATABILITY] Saved LSTM dataset: {output}")
    return output

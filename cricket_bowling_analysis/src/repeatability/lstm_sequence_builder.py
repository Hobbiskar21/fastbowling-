"""Convert one delivery movement-curve CSV into a 120 timestep LSTM sample."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, LSTM_FEATURES, OUTPUT_SUBDIRS, PHASE_FIXED_LENGTHS, PHASE_NAMES
from .feature_repair import repair_repeatability_features


def build_lstm_sequence_from_curve_csv(
    curve_csv_path,
    output_csv_path: Optional[str] = None,
    output_npy_path: Optional[str] = None,
    metadata_output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build and save one delivery sequence: 120 x len(LSTM_FEATURES)."""
    input_path = Path(curve_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Curve CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df, repaired = repair_repeatability_features(df)
    if repaired:
        print(f"[REPEATABILITY] Repaired/created {len(repaired)} feature columns before LSTM sequence.")
    ordered = []
    for phase in PHASE_NAMES:
        phase_df = df[df["phase"] == phase].sort_values("normalized_index").copy()
        expected = PHASE_FIXED_LENGTHS[phase]
        if len(phase_df) != expected:
            print(f"[REPEATABILITY] Warning: {phase} has {len(phase_df)} rows, expected {expected}.")
        ordered.append(phase_df.head(expected))
    seq_df = pd.concat(ordered, ignore_index=True) if ordered else pd.DataFrame()

    for feature in LSTM_FEATURES:
        if feature not in seq_df.columns:
            seq_df[feature] = 0.0

    feature_df = seq_df[LSTM_FEATURES].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)
    sequence = feature_df.to_numpy(dtype=np.float32)
    expected_rows = sum(PHASE_FIXED_LENGTHS.values())
    if sequence.shape != (expected_rows, len(LSTM_FEATURES)):
        raise ValueError(f"Bad LSTM sequence shape {sequence.shape}; expected {(expected_rows, len(LSTM_FEATURES))}")

    delivery_id = str(seq_df["delivery_id"].iloc[0]) if "delivery_id" in seq_df.columns and not seq_df.empty else input_path.stem
    bowler_id = str(seq_df["bowler_id"].iloc[0]) if "bowler_id" in seq_df.columns and not seq_df.empty else "unknown_bowler"
    out_dir = Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["delivery_sequences"]
    csv_path = Path(output_csv_path) if output_csv_path else out_dir / f"{bowler_id}_{delivery_id}_sequence.csv"
    npy_path = Path(output_npy_path) if output_npy_path else out_dir / f"{bowler_id}_{delivery_id}_sequence.npy"
    meta_path = Path(metadata_output_path) if metadata_output_path else out_dir / f"{bowler_id}_{delivery_id}_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, sequence)
    feature_df.to_csv(csv_path, index=False)
    metadata = pd.DataFrame([{
        "bowler_id": bowler_id,
        "delivery_id": delivery_id,
        "sequence_rows": sequence.shape[0],
        "sequence_features": sequence.shape[1],
        "features": ",".join(LSTM_FEATURES),
    }])
    metadata.to_csv(meta_path, index=False)
    print(f"[REPEATABILITY] Saved LSTM sequence CSV: {csv_path}")
    print(f"[REPEATABILITY] Saved LSTM sequence NPY: {npy_path}")
    print(f"[REPEATABILITY] Saved sequence metadata: {meta_path}")
    return feature_df, sequence, metadata

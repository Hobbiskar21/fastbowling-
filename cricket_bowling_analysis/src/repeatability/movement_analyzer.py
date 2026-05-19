"""Build fixed-length movement curves from confirmed phase-labelled frames."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, OUTPUT_SUBDIRS, PHASE_FEATURES, PHASE_FIXED_LENGTHS, PHASE_ID_MAP, PHASE_NAMES
from .feature_repair import repair_repeatability_features


def _interp(values: pd.Series, target_len: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(arr) == 0:
        return np.zeros(target_len)
    if np.isnan(arr).all():
        return np.zeros(target_len)
    series = pd.Series(arr).ffill().bfill().fillna(0).to_numpy(dtype=float)
    if len(series) == 1:
        return np.repeat(series[0], target_len)
    return np.interp(np.linspace(0, len(series) - 1, target_len), np.arange(len(series)), series)


def build_movement_curves(
    phase_labelled_csv_path,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Resample each of the 7 phases to its configured fixed length."""
    input_path = Path(phase_labelled_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Phase-labelled CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df, repaired = repair_repeatability_features(df)
    if repaired:
        print(f"[REPEATABILITY] Repaired/created {len(repaired)} feature columns before movement curves.")
    if "delivery_id" not in df.columns:
        df["delivery_id"] = input_path.stem
    if "bowler_id" not in df.columns:
        df["bowler_id"] = "unknown_bowler"

    rows = []
    for delivery_id, delivery_df in df.groupby("delivery_id", dropna=False):
        bowler_id = str(delivery_df["bowler_id"].iloc[0])
        for phase in PHASE_NAMES:
            phase_df = delivery_df[delivery_df["phase"] == phase].sort_values("frame_id")
            target_len = PHASE_FIXED_LENGTHS[phase]
            features = PHASE_FEATURES[phase]

            for idx in range(target_len):
                rows.append({
                    "bowler_id": bowler_id,
                    "delivery_id": delivery_id,
                    "phase": phase,
                    "phase_id": PHASE_ID_MAP[phase],
                    "normalized_index": idx,
                    "normalized_time": idx / max(target_len - 1, 1),
                    "normalized_phase_time": idx / max(target_len - 1, 1),
                })
            start = len(rows) - target_len
            if phase_df.empty:
                continue
            for feature in features:
                vals = _interp(phase_df[feature], target_len)
                for offset, value in enumerate(vals):
                    rows[start + offset][feature] = value

    curves_df = pd.DataFrame(rows)
    feature_columns = sorted(set().union(*(set(v) for v in PHASE_FEATURES.values())))
    for feature in feature_columns:
        if feature not in curves_df.columns:
            curves_df[feature] = 0.0
        curves_df[feature] = pd.to_numeric(curves_df[feature], errors="coerce").ffill().bfill().fillna(0.0)
    delivery_id = str(df["delivery_id"].iloc[0])
    out_path = Path(output_csv_path) if output_csv_path else Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["movement_curves"] / f"{delivery_id}_movement_curves.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curves_df.to_csv(out_path, index=False)
    print(f"[REPEATABILITY] Saved movement curves: {out_path}")
    return curves_df

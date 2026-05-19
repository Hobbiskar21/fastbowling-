"""Run repeatability preprocessing and LSTM inference for one delivery CSV."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_OUTPUT_DIR, LSTM_FEATURES, OUTPUT_SUBDIRS, PHASE_NAMES
from .lstm_model import BowlingRepeatabilityLSTM
from .output_dashboard import create_repeatability_dashboard, verdict_from_score
from .repeatability_visualizer import plot_lstm_sequence_features
from .pipeline import create_output_dirs, process_csv


def _load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = BowlingRepeatabilityLSTM(input_size=int(checkpoint.get("input_size", len(LSTM_FEATURES))))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def test_single_delivery(
    frame_csv_path,
    model_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    manual_events: Optional[dict] = None,
    review_phases: bool = False,
):
    """Preprocess one frame CSV, run model prediction, and save dashboard."""
    output = Path(output_dir)
    dirs = create_output_dirs(output_dir)
    paths = process_csv(frame_csv_path, dirs, manual_events=manual_events)
    sequence = np.load(paths["sequence_npy"]).astype(np.float32)
    metadata = pd.read_csv(paths["metadata_csv"])

    model = _load_model(Path(model_path))
    with torch.no_grad():
        overall, phases = model(torch.tensor(sequence[None, :, :], dtype=torch.float32))
    overall_score = float(overall.squeeze().item() * 100)
    phase_scores = {phase: float(value * 100) for phase, value in zip(PHASE_NAMES, phases.squeeze().numpy())}
    strongest = max(phase_scores, key=phase_scores.get)
    weakest = min(phase_scores, key=phase_scores.get)
    bowler_id = str(metadata.iloc[0]["bowler_id"])
    delivery_id = str(metadata.iloc[0]["delivery_id"])

    pred_path = output / OUTPUT_SUBDIRS["predictions"] / f"{bowler_id}_{delivery_id}_prediction.csv"
    row = {
        "bowler_id": bowler_id,
        "delivery_id": delivery_id,
        "overall_score": overall_score,
        "verdict": verdict_from_score(overall_score),
        "strongest_phase": strongest,
        "weakest_phase": weakest,
    }
    row.update({f"{phase}_score": score for phase, score in phase_scores.items()})
    pd.DataFrame([row]).to_csv(pred_path, index=False)
    dashboard_path = output / OUTPUT_SUBDIRS["dashboard"] / f"{bowler_id}_{delivery_id}_dashboard.png"
    create_repeatability_dashboard(bowler_id, delivery_id, overall_score, phase_scores, dashboard_path)
    graph_path = output / OUTPUT_SUBDIRS["graphs"] / f"{bowler_id}_{delivery_id}_lstm_sequence.png"
    plot_lstm_sequence_features(paths["sequence_csv"], graph_path)
    print(f"[REPEATABILITY] Saved prediction: {pred_path}")
    return row

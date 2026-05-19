"""Reusable side-on repeatability preprocessing pipeline."""

import shutil
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, OUTPUT_SUBDIRS
from .lstm_sequence_builder import build_lstm_sequence_from_curve_csv
from .movement_analyzer import build_movement_curves
from .phase_segmenter import assign_phases_to_frames
from .repeatability_input_builder import build_repeatability_input
from .sideon_feature_selector import select_sideon_features
from .sideon_phase_detector import detect_sideon_phases


def create_output_dirs(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Create and return all repeatability output directories."""
    root = Path(output_dir)
    dirs = {}
    for key, name in OUTPUT_SUBDIRS.items():
        dirs[key] = root / name
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def copy_auto_to_confirmed(auto_phases, auto_events, confirmed_phases, confirmed_events) -> None:
    """Use automatic phase detection as confirmed input for downstream steps."""
    confirmed_phases.parent.mkdir(parents=True, exist_ok=True)
    confirmed_events.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(auto_phases, confirmed_phases)
    shutil.copyfile(auto_events, confirmed_events)
    print(f"[REPEATABILITY] Auto phases confirmed: {confirmed_phases}")
    print(f"[REPEATABILITY] Auto events confirmed: {confirmed_events}")


def process_csv(frame_csv_path, dirs: dict, manual_events: dict | None = None) -> dict:
    """Run repeatability preprocessing from frame CSV through LSTM sequence."""
    source = Path(frame_csv_path)
    delivery_id = source.stem

    repeat_input = dirs["input"] / f"{delivery_id}_repeatability_input.csv"
    build_repeatability_input(source, repeat_input)

    auto_phases = dirs["auto_phases"] / f"{delivery_id}_auto_phases.csv"
    auto_events = dirs["auto_events"] / f"{delivery_id}_auto_events.csv"
    detect_sideon_phases(repeat_input, auto_phases, auto_events, manual_events=manual_events)

    confirmed_phases = dirs["confirmed_phases"] / f"{delivery_id}_confirmed_phases.csv"
    confirmed_events = dirs["confirmed_events"] / f"{delivery_id}_confirmed_events.csv"
    copy_auto_to_confirmed(auto_phases, auto_events, confirmed_phases, confirmed_events)

    features = dirs["sideon_features"] / f"{delivery_id}_sideon_features.csv"
    report = dirs["sideon_features"] / f"{delivery_id}_selected_features_report.csv"
    select_sideon_features(repeat_input, confirmed_phases, features, report)

    labelled = dirs["phase_labelled"] / f"{delivery_id}_phase_labelled.csv"
    assign_phases_to_frames(features, confirmed_phases, labelled)

    curves = dirs["movement_curves"] / f"{delivery_id}_movement_curves.csv"
    build_movement_curves(labelled, curves)

    sequence_csv = dirs["delivery_sequences"] / f"{delivery_id}_sequence.csv"
    sequence_npy = dirs["delivery_sequences"] / f"{delivery_id}_sequence.npy"
    metadata_csv = dirs["delivery_sequences"] / f"{delivery_id}_metadata.csv"
    build_lstm_sequence_from_curve_csv(curves, sequence_csv, sequence_npy, metadata_csv)

    return {
        "repeat_input": repeat_input,
        "auto_phases": auto_phases,
        "auto_events": auto_events,
        "confirmed_phases": confirmed_phases,
        "confirmed_events": confirmed_events,
        "features": features,
        "phase_labelled": labelled,
        "curves": curves,
        "sequence_csv": sequence_csv,
        "sequence_npy": sequence_npy,
        "metadata_csv": metadata_csv,
    }

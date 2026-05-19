"""Interactive review tool for suggested side-on repeatability phases."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, EVENT_NAMES, OUTPUT_SUBDIRS
from .sideon_phase_detector import build_phases_from_events


def _default_paths(delivery_id: str) -> Tuple[Path, Path]:
    root = Path(DEFAULT_OUTPUT_DIR)
    return (
        root / OUTPUT_SUBDIRS["confirmed_phases"] / f"{delivery_id}_confirmed_phases.csv",
        root / OUTPUT_SUBDIRS["confirmed_events"] / f"{delivery_id}_confirmed_events.csv",
    )


def _events_to_dict(events_df: pd.DataFrame) -> dict:
    return {
        str(row["event_name"]): int(row["frame_id"])
        for _, row in events_df.iterrows()
        if pd.notna(row.get("frame_id"))
    }


def _print_review(events_df: pd.DataFrame, phases_df: pd.DataFrame) -> None:
    print("\n[REPEATABILITY] Suggested events")
    for _, row in events_df.iterrows():
        print(f"  {row['event_name']}: {row['frame_id']} ({row.get('detection_method', '')})")
    print("\n[REPEATABILITY] Suggested phases")
    for _, row in phases_df.iterrows():
        print(f"  {row['phase']}: {row['start_frame']} -> {row['end_frame']} | event={row.get('event_frame')}")


def _ask_yes_no(prompt: str, default: str = "n") -> bool:
    suffix = " [Y/n]: " if default.lower() == "y" else " [y/N]: "
    value = input(prompt + suffix).strip().lower()
    if not value:
        value = default.lower()
    return value.startswith("y")


def _manual_events_from_user(current_events: dict) -> dict:
    manual = {}
    print("\nEnter corrected event frames. Press Enter to keep suggested value.")
    for event_name in EVENT_NAMES:
        suggested = current_events.get(event_name, "")
        raw = input(f"{event_name} [{suggested}]: ").strip()
        manual[event_name] = int(raw) if raw else int(suggested)
    return manual


def review_and_confirm_phases(
    repeatability_input_csv_path,
    auto_phase_csv_path,
    auto_events_csv_path,
    confirmed_phase_csv_path: Optional[str] = None,
    confirmed_events_csv_path: Optional[str] = None,
    allow_overwrite: bool = False,
    review: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Review auto events/phases and save confirmed CSVs.

    If review is False, existing confirmed files are reused; otherwise auto
    suggestions are copied to confirmed files.
    """
    input_df = pd.read_csv(repeatability_input_csv_path)
    phases_df = pd.read_csv(auto_phase_csv_path)
    events_df = pd.read_csv(auto_events_csv_path)

    delivery_id = str(input_df["delivery_id"].iloc[0]) if "delivery_id" in input_df.columns else Path(repeatability_input_csv_path).stem
    default_phase_path, default_events_path = _default_paths(delivery_id)
    confirmed_phase_path = Path(confirmed_phase_csv_path) if confirmed_phase_csv_path else default_phase_path
    confirmed_events_path = Path(confirmed_events_csv_path) if confirmed_events_csv_path else default_events_path

    if confirmed_phase_path.exists() and confirmed_events_path.exists() and not allow_overwrite:
        if not review or not _ask_yes_no("Confirmed phases already exist for this delivery. Overwrite?", default="n"):
            print(f"[REPEATABILITY] Using existing confirmed phases: {confirmed_phase_path}")
            return pd.read_csv(confirmed_phase_path), pd.read_csv(confirmed_events_path)

    _print_review(events_df, phases_df)
    events = _events_to_dict(events_df)
    if review and not _ask_yes_no("Accept detected phases?", default="y"):
        events = _manual_events_from_user(events)
        min_frame = int(input_df["frame_id"].min())
        max_frame = int(input_df["frame_id"].max())
        phases_df = build_phases_from_events(events, min_frame, max_frame, delivery_id)
        events_df = pd.DataFrame([
            {
                "delivery_id": delivery_id,
                "event_name": name,
                "frame_id": events[name],
                "confidence": 1.0,
                "detection_method": "manual_review",
            }
            for name in EVENT_NAMES
        ])

    confirmed_phase_path.parent.mkdir(parents=True, exist_ok=True)
    confirmed_events_path.parent.mkdir(parents=True, exist_ok=True)
    phases_df.to_csv(confirmed_phase_path, index=False)
    events_df.to_csv(confirmed_events_path, index=False)
    print(f"[REPEATABILITY] Saved confirmed phases: {confirmed_phase_path}")
    print(f"[REPEATABILITY] Saved confirmed events: {confirmed_events_path}")
    return phases_df, events_df

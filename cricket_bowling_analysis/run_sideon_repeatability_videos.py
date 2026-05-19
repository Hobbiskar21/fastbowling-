"""Automatic side-on repeatability runner for raw videos in a separate folder.

This file creates annotated pose videos and frame CSVs, then starts the
7-phase repeatability pipeline from those CSVs.

Default input folders:
    input_videos/repeatability/train
    input_videos/repeatability/test
"""

import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path

import cv2
import pandas as pd

from src.repeatability.config import DEFAULT_OUTPUT_DIR, LSTM_FEATURES, OUTPUT_SUBDIRS, PHASE_DISPLAY_NAMES, PHASE_FEATURES, PHASE_NAMES, SIDEON_FEATURES
from src.repeatability.pipeline import create_output_dirs, process_csv
from src.repeatability.video_analysis import run_repeatability_video_analysis


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
REQUIRED_REPEATABILITY_COLUMNS = {
    "frame_id",
    "wrist_speed",
    "hip_speed",
    "front_ankle_y",
    "back_ankle_y",
    "bowling_wrist_y",
    "hip_center_y",
}
REQUIRED_TRAINING_COLUMNS = set(SIDEON_FEATURES).union(LSTM_FEATURES).union(*(PHASE_FEATURES[p] for p in PHASE_FEATURES))


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_video_dir(split: str) -> Path:
    return workspace_root() / "input_videos" / "repeatability" / split


def resolve_output_dir(output_dir: str) -> Path:
    """Resolve repeatability outputs against the shared workspace root."""
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return workspace_root() / path


def find_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Repeatability video folder not found: {video_dir}")
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def find_videos_with_fallback(video_dir: Path, split: str) -> list[Path]:
    """Find split videos, falling back to input_videos/repeatability for train."""
    videos = find_videos(video_dir)
    if videos or split != "train":
        return videos

    root_dir = video_dir.parent
    root_videos = find_videos(root_dir)
    if root_videos:
        print(f"[REPEATABILITY] No videos in {video_dir}; using videos directly in {root_dir} as train videos.")
    return root_videos


def analysis_paths(analysis_root: Path, video_path: Path) -> tuple[Path, Path, Path]:
    session_id = f"single_{video_path.stem}"
    frame_csv = analysis_root / "artifacts" / f"{session_id}_frame_data.csv"
    analyzed_video = analysis_root / "videos" / f"{session_id}_analysis.mp4"
    repeatability_marker = analysis_root / "artifacts" / f"{session_id}_repeatability_only.txt"
    return frame_csv, analyzed_video, repeatability_marker


def manifest_path(video_path: Path, split: str, output_dir: Path) -> Path:
    return output_dir / "video_analysis" / "_manifests" / split / f"{video_path.stem}_files.json"


def source_info_path(video_path: Path, split: str, output_dir: Path) -> Path:
    return output_dir / "video_analysis" / "_manifests" / split / f"{video_path.stem}_source.json"


def source_signature(video_path: Path) -> dict:
    stat = video_path.stat()
    return {
        "path": str(video_path.resolve()),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def source_matches_manifest(video_path: Path, split: str, output_dir: Path) -> bool:
    path = source_info_path(video_path, split, output_dir)
    if not path.exists():
        return False
    try:
        previous = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return previous == source_signature(video_path)


def save_source_manifest(video_path: Path, split: str, output_dir: Path) -> None:
    path = source_info_path(video_path, split, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source_signature(video_path), indent=2))


def load_artifact_manifest(video_stem: str, split: str, output_dir: Path) -> list[Path]:
    path = output_dir / "video_analysis" / "_manifests" / split / f"{video_stem}_files.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return [Path(item) for item in payload.get("files", [])]


def save_artifact_manifest(video_path: Path, split: str, output_dir: Path, files: list[Path]) -> None:
    path = manifest_path(video_path, split, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video_path.resolve()),
        "files": [str(file_path) for file_path in files],
    }
    path.write_text(json.dumps(payload, indent=2))


def delivery_id_for_video(video_path: Path) -> str:
    return f"single_{video_path.stem}_frame_data"


def session_id_for_video(video_path: Path) -> str:
    return f"single_{video_path.stem}"


def expected_artifacts(video_path: Path, split: str, output_dir: Path) -> list[Path]:
    delivery_id = delivery_id_for_video(video_path)
    session_id = session_id_for_video(video_path)
    analysis_root = output_dir / "video_analysis" / split / video_path.stem
    return [
        analysis_root / "artifacts" / f"{session_id}_frame_data.csv",
        analysis_root / "artifacts" / f"{session_id}_repeatability_only.txt",
        analysis_root / "videos" / f"{session_id}_analysis.mp4",
        analysis_root / "videos" / f"{session_id}_analysis_7_phase_analysis.mp4",
        manifest_path(video_path, split, output_dir),
        source_info_path(video_path, split, output_dir),
        output_dir / OUTPUT_SUBDIRS["input"] / f"{delivery_id}_repeatability_input.csv",
        output_dir / OUTPUT_SUBDIRS["auto_phases"] / f"{delivery_id}_auto_phases.csv",
        output_dir / OUTPUT_SUBDIRS["auto_events"] / f"{delivery_id}_auto_events.csv",
        output_dir / OUTPUT_SUBDIRS["confirmed_phases"] / f"{delivery_id}_confirmed_phases.csv",
        output_dir / OUTPUT_SUBDIRS["confirmed_events"] / f"{delivery_id}_confirmed_events.csv",
        output_dir / OUTPUT_SUBDIRS["sideon_features"] / f"{delivery_id}_sideon_features.csv",
        output_dir / OUTPUT_SUBDIRS["sideon_features"] / f"{delivery_id}_selected_features_report.csv",
        output_dir / OUTPUT_SUBDIRS["phase_labelled"] / f"{delivery_id}_phase_labelled.csv",
        output_dir / OUTPUT_SUBDIRS["movement_curves"] / f"{delivery_id}_movement_curves.csv",
        output_dir / OUTPUT_SUBDIRS["delivery_sequences"] / f"{delivery_id}_sequence.csv",
        output_dir / OUTPUT_SUBDIRS["delivery_sequences"] / f"{delivery_id}_sequence.npy",
        output_dir / OUTPUT_SUBDIRS["delivery_sequences"] / f"{delivery_id}_metadata.csv",
        output_dir / "phase_detection_videos" / split / f"{video_path.stem}_7_phase_detection.mp4",
    ]


def artifacts_complete(video_path: Path, split: str, output_dir: Path) -> bool:
    required = expected_artifacts(video_path, split, output_dir)
    if not all(path.exists() for path in required):
        return False
    frame_csv = output_dir / "video_analysis" / split / video_path.stem / "artifacts" / f"{session_id_for_video(video_path)}_frame_data.csv"
    if not is_valid_repeatability_csv(frame_csv):
        return False
    if not source_matches_manifest(video_path, split, output_dir):
        print(f"[REPEATABILITY] Source video changed or has no fingerprint; rebuilding: {video_path.name}")
        return False
    try:
        video_mtime = video_path.stat().st_mtime
    except OSError:
        return False
    # Rebuild if the source video is newer than any derived artifact.
    return all(path.stat().st_mtime >= video_mtime for path in required)


def _remove_path(path: Path) -> None:
    def _make_writable(func, target, _exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except PermissionError:
            raise

    if path.is_dir():
        for attempt in range(3):
            try:
                shutil.rmtree(path, onerror=_make_writable)
                print(f"[REPEATABILITY] Removed stale folder: {path}")
                return
            except PermissionError as exc:
                if attempt == 2:
                    print(f"[REPEATABILITY] Warning: could not remove locked stale folder: {path} ({exc})")
                    return
                time.sleep(0.5)
    elif path.exists():
        for attempt in range(3):
            try:
                os.chmod(path, stat.S_IWRITE)
                path.unlink()
                print(f"[REPEATABILITY] Removed stale file: {path}")
                return
            except PermissionError as exc:
                if attempt == 2:
                    print(f"[REPEATABILITY] Warning: could not remove locked stale file: {path} ({exc})")
                    return
                time.sleep(0.5)


def prune_removed_video_artifacts(active_videos: list[Path], split: str, output_dir: Path) -> None:
    active_stems = {video.stem for video in active_videos}
    active_delivery_ids = {delivery_id_for_video(video) for video in active_videos}

    analysis_split_dir = output_dir / "video_analysis" / split
    if analysis_split_dir.exists():
        for child in analysis_split_dir.iterdir():
            if child.is_dir() and child.name not in active_stems:
                _remove_path(child)

    manifest_dir = output_dir / "video_analysis" / "_manifests" / split
    if manifest_dir.exists():
        for manifest in manifest_dir.glob("*_files.json"):
            video_stem = manifest.name.replace("_files.json", "")
            if video_stem not in active_stems:
                for old_path in load_artifact_manifest(video_stem, split, output_dir):
                    _remove_path(old_path)
                _remove_path(manifest)
                _remove_path(source_info_path(Path(video_stem), split, output_dir))

    phase_video_dir = output_dir / "phase_detection_videos" / split
    if phase_video_dir.exists():
        for file_path in phase_video_dir.glob("*_7_phase_detection.mp4"):
            video_stem = file_path.name.replace("_7_phase_detection.mp4", "")
            if video_stem not in active_stems:
                _remove_path(file_path)

    for subdir_name in OUTPUT_SUBDIRS.values():
        folder = output_dir / subdir_name
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
            matched_current_video = any(file_path.name.startswith(delivery_id) for delivery_id in active_delivery_ids)
            if file_path.name.startswith("single_") and not matched_current_video:
                _remove_path(file_path)


def phase_for_frame(phases_df: pd.DataFrame, frame_id: int) -> str:
    matches = phases_df[(phases_df["start_frame"] <= frame_id) & (phases_df["end_frame"] >= frame_id)]
    if matches.empty:
        return "unassigned"
    return str(matches.iloc[0]["phase"])


def is_valid_repeatability_csv(frame_csv: Path) -> bool:
    if not frame_csv.exists():
        return False
    try:
        columns = set(pd.read_csv(frame_csv, nrows=0).columns)
    except Exception:
        return False
    missing = sorted((REQUIRED_REPEATABILITY_COLUMNS | REQUIRED_TRAINING_COLUMNS) - columns)
    if missing:
        print(f"[REPEATABILITY] Existing CSV missing repeatability columns; regenerating: {', '.join(missing)}")
        return False
    return True


def phase_index(phase: str) -> int:
    try:
        return PHASE_NAMES.index(phase) + 1
    except ValueError:
        return 0


def print_phase_summary(confirmed_phase_csv_path: Path) -> pd.DataFrame:
    phases_df = pd.read_csv(confirmed_phase_csv_path)
    print("[REPEATABILITY] 7 detected phases:")
    for _, row in phases_df.iterrows():
        phase = str(row["phase"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        event = row.get("event_frame", "")
        event_text = "" if pd.isna(event) or event == "" else f", event={int(event)}"
        label = PHASE_DISPLAY_NAMES.get(phase, phase)
        print(f"  {phase_index(phase)}. {label}: frames {start}-{end}{event_text}")
    return phases_df


def draw_phase_label(frame, phase: str, frame_id: int):
    out = frame.copy()
    label = PHASE_DISPLAY_NAMES.get(phase, phase)
    idx = phase_index(phase)
    title = f"{idx}/7 {label}" if idx else "Unassigned"
    cv2.putText(out, title, (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, f"Frame {frame_id}", (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def render_phase_detection_video(analyzed_video_path: Path, confirmed_phase_csv_path: Path, output_path: Path) -> Path:
    """Render an analyzed video with phase labels and 1s pauses at key events."""
    phases_df = pd.read_csv(confirmed_phase_csv_path)
    pause_event_frames = {
        int(row["event_frame"])
        for _, row in phases_df.iterrows()
        if str(row["phase"]) in {"jump_bound", "bfc_window", "ffc_window", "ffc_to_release"}
        and pd.notna(row.get("event_frame"))
    }
    min_phase_frame = int(phases_df["start_frame"].min())
    max_phase_frame = int(phases_df["end_frame"].max())

    cap = cv2.VideoCapture(str(analyzed_video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open analyzed video: {analyzed_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pause_frames = max(1, int(round(fps * 1.0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open writer: {output_path}")

    video_frame_id = 0
    paused_events = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total_frames > 1:
            frame_id = int(round(min_phase_frame + (video_frame_id / (total_frames - 1)) * (max_phase_frame - min_phase_frame)))
        else:
            frame_id = video_frame_id
        phase = phase_for_frame(phases_df, frame_id)
        labelled = draw_phase_label(frame, phase, frame_id)
        pause_frame = next((event for event in pause_event_frames if abs(frame_id - event) <= 1), None)
        if pause_frame is not None and pause_frame not in paused_events:
            for _ in range(pause_frames):
                writer.write(labelled)
            paused_events.add(pause_frame)
        writer.write(labelled)
        video_frame_id += 1

    cap.release()
    writer.release()
    print(f"[REPEATABILITY] Saved phase detection video: {output_path}")
    return output_path


def process_video(video_path: Path, split: str, output_dir: Path, force_analysis: bool = False) -> None:
    analysis_root = output_dir / "video_analysis" / split / video_path.stem
    if not force_analysis and artifacts_complete(video_path, split, output_dir):
        print(f"[REPEATABILITY] Already processed, skipping: {video_path.name}")
        return
    if force_analysis:
        print(f"[REPEATABILITY] Force enabled; rebuilding: {video_path.name}")
        for old_path in load_artifact_manifest(video_path.stem, split, output_dir):
            _remove_path(old_path)
        for path in expected_artifacts(video_path, split, output_dir):
            _remove_path(path)
        if analysis_root.exists():
            _remove_path(analysis_root)

    frame_csv, analyzed_video, repeatability_marker = analysis_paths(analysis_root, video_path)
    valid_csv = is_valid_repeatability_csv(frame_csv)
    if not valid_csv or not analyzed_video.exists() or not repeatability_marker.exists():
        print(f"[REPEATABILITY] Running repeatability video analysis: {video_path}")
        if analysis_root.exists():
            shutil.rmtree(analysis_root)
        run_repeatability_video_analysis(str(video_path), str(analysis_root))
        frame_csv, analyzed_video, repeatability_marker = analysis_paths(analysis_root, video_path)
    else:
        print(f"[REPEATABILITY] Reusing existing analyzed video/CSV for: {video_path.name}")

    if not frame_csv.exists():
        raise FileNotFoundError(
            f"Expected frame CSV not found: {frame_csv}\n"
            f"Repeatability video analysis should have written it under: {analysis_root / 'artifacts'}"
        )
    if not analyzed_video.exists():
        raise FileNotFoundError(f"Expected analyzed video not found: {analyzed_video}")

    dirs = create_output_dirs(str(output_dir))
    print("[REPEATABILITY] Running 7-phase repeatability pipeline.")
    outputs = process_csv(frame_csv, dirs)
    confirmed_phase_csv = outputs["confirmed_phases"]
    print_phase_summary(confirmed_phase_csv)
    phase_video_path = output_dir / "phase_detection_videos" / split / f"{video_path.stem}_7_phase_detection.mp4"
    render_phase_detection_video(analyzed_video, confirmed_phase_csv, phase_video_path)
    analysis_phase_video = analyzed_video.with_name(f"{analyzed_video.stem}_7_phase_analysis.mp4")
    render_phase_detection_video(analyzed_video, confirmed_phase_csv, analysis_phase_video)
    save_source_manifest(video_path, split, output_dir)
    save_artifact_manifest(video_path, split, output_dir, expected_artifacts(video_path, split, output_dir))


def main():
    parser = argparse.ArgumentParser(description="Run automatic side-on repeatability from raw videos.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Input split under input_videos/repeatability.")
    parser.add_argument("--video_dir", default=None, help="Optional explicit folder of side-on videos.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force_analysis", action="store_true", help="Regenerate per-video analysis outputs.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir) if args.video_dir else default_video_dir(args.split)
    output_dir = resolve_output_dir(args.output_dir)
    print(f"[REPEATABILITY] Output root: {output_dir}")
    videos = find_videos_with_fallback(video_dir, args.split)
    prune_removed_video_artifacts(videos, args.split, output_dir)
    if not videos:
        print(f"[REPEATABILITY] No videos found in: {video_dir}")
        return

    print(f"[REPEATABILITY] Processing {len(videos)} {args.split} video(s) from {video_dir}")
    for video in videos:
        process_video(video, args.split, output_dir, force_analysis=args.force_analysis)
    print(f"[REPEATABILITY] Done. Output root: {output_dir}")


if __name__ == "__main__":
    main()

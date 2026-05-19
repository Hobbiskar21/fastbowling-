"""Score repeatability for one side-on bowling video.

Input video -> repeatability frame CSV -> 7 phases -> LSTM sequence -> score.
"""

import argparse
import json
import os
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.repeatability.config import DEFAULT_OUTPUT_DIR, LSTM_FEATURES, OUTPUT_SUBDIRS, PHASE_NAMES
from src.repeatability.output_dashboard import create_repeatability_dashboard, verdict_from_score
from src.repeatability.pipeline import create_output_dirs, process_csv
from src.repeatability.repeatability_visualizer import plot_lstm_sequence_features
from src.repeatability.video_analysis import run_repeatability_video_analysis
from run_sideon_repeatability_videos import render_phase_detection_video

try:
    import torch
    from src.repeatability.lstm_model import BowlingRepeatabilityLSTM
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "[REPEATABILITY] PyTorch is not installed in this Python environment.\n"
            "Install project dependencies first:\n"
            "  python -m pip install -r requirements.txt\n"
            "Then train/score again."
        ) from exc
    raise


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return workspace_root() / candidate


def default_video_dir() -> Path:
    return workspace_root() / "input_videos" / "repeatability"


def list_videos(video_dir: Path, recursive: bool = True) -> list[Path]:
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    if not video_dir.exists():
        return []
    pattern = "**/*" if recursive else "*"
    videos = sorted(path for path in video_dir.glob(pattern) if path.is_file() and path.suffix.lower() in extensions)
    if not videos and not recursive:
        videos = sorted(path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    return videos


def choose_video_interactive(video_dir: Path, recursive: bool = True) -> Path:
    videos = list_videos(video_dir, recursive=recursive)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {video_dir}")

    print(f"\n[REPEATABILITY] Available videos in {video_dir}:")
    for idx, video in enumerate(videos, 1):
        try:
            label = video.relative_to(video_dir)
        except ValueError:
            label = video
        print(f"  {idx}. {label}")

    while True:
        try:
            choice = input(f"\nSelect video (1-{len(videos)}): ").strip()
        except EOFError as exc:
            raise SystemExit(
                "[REPEATABILITY] No interactive input was available.\n"
                "Run this command in PowerShell/CMD and type the number, or pass a video path:\n"
                "  python score_sideon_repeatability_video.py C:\\path\\to\\video.mp4"
            ) from exc
        try:
            selected = int(choice) - 1
        except ValueError:
            print("[REPEATABILITY] Please enter a number.")
            continue
        if 0 <= selected < len(videos):
            return videos[selected]
        print(f"[REPEATABILITY] Please enter a number between 1 and {len(videos)}.")


def remove_if_exists(path: Path) -> None:
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
                print(f"[REPEATABILITY] Removed old folder: {path}")
                return
            except PermissionError as exc:
                if attempt == 2:
                    print(f"[REPEATABILITY] Warning: could not remove locked old folder: {path} ({exc})")
                    return
                time.sleep(0.5)
    elif path.exists():
        for attempt in range(3):
            try:
                os.chmod(path, stat.S_IWRITE)
                path.unlink()
                print(f"[REPEATABILITY] Removed old file: {path}")
                return
            except PermissionError as exc:
                if attempt == 2:
                    print(f"[REPEATABILITY] Warning: could not remove locked old file: {path} ({exc})")
                    return
                time.sleep(0.5)


def manifest_path(video_path: Path, output_dir: Path) -> Path:
    return output_dir / "video_scoring" / "_manifests" / f"{video_path.stem}_files.json"


def load_manifest(video_path: Path, output_dir: Path) -> list[Path]:
    path = manifest_path(video_path, output_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return [Path(item) for item in data.get("files", [])]


def save_manifest(video_path: Path, output_dir: Path, files: list[Path]) -> None:
    path = manifest_path(video_path, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video": str(video_path), "files": [str(file_path) for file_path in files]}
    path.write_text(json.dumps(payload, indent=2))
    print(f"[REPEATABILITY] Saved scoring manifest: {path}")


def cleanup_empty_parents(paths: list[Path], output_dir: Path) -> None:
    protected = output_dir.resolve()
    for path in sorted({p.parent for p in paths}, key=lambda p: len(str(p)), reverse=True):
        current = path
        while current != protected and protected in current.resolve().parents:
            try:
                current.rmdir()
                print(f"[REPEATABILITY] Removed empty folder: {current}")
            except OSError:
                break
            current = current.parent


def clear_previous_score_outputs(video_path: Path, output_dir: Path) -> None:
    session_id = f"single_{video_path.stem}"
    delivery_id = f"{session_id}_frame_data"
    print(f"[REPEATABILITY] Clearing old scoring outputs for: {video_path.name}")

    manifest_files = load_manifest(video_path, output_dir)
    for old_file in manifest_files:
        remove_if_exists(old_file)

    targets = [
        output_dir / "video_scoring" / video_path.stem,
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
        output_dir / OUTPUT_SUBDIRS["predictions"] / f"{delivery_id}_prediction.csv",
        output_dir / OUTPUT_SUBDIRS["dashboard"] / f"{delivery_id}_dashboard.png",
        output_dir / OUTPUT_SUBDIRS["graphs"] / f"{delivery_id}_lstm_sequence.png",
    ]
    for target in targets:
        remove_if_exists(target)
    remove_if_exists(manifest_path(video_path, output_dir))
    cleanup_empty_parents(manifest_files + targets, output_dir)


def load_model(model_path: Path) -> BowlingRepeatabilityLSTM:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained repeatability model not found: {model_path}\n"
            "Train it first with train_sideon_lstm.py, then pass --model_path."
        )
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    model = BowlingRepeatabilityLSTM(input_size=int(checkpoint.get("input_size", len(LSTM_FEATURES))))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_video(video_path: Path, model_path: Path, output_dir: Path, force: bool = False) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = load_model(model_path)
    clear_previous_score_outputs(video_path, output_dir)
    analysis_root = output_dir / "video_scoring" / video_path.stem

    session_id = f"single_{video_path.stem}"
    frame_csv = analysis_root / "artifacts" / f"{session_id}_frame_data.csv"
    print(f"[REPEATABILITY] Building repeatability video artifacts: {video_path}")
    video_artifacts = run_repeatability_video_analysis(str(video_path), str(analysis_root))

    dirs = create_output_dirs(str(output_dir))
    paths = process_csv(frame_csv, dirs)
    sequence = np.load(paths["sequence_npy"]).astype(np.float32)
    metadata = pd.read_csv(paths["metadata_csv"])

    with torch.no_grad():
        overall, phases = model(torch.tensor(sequence[None, :, :], dtype=torch.float32))

    overall_score = float(overall.squeeze().item() * 100)
    phase_scores = {phase: float(value * 100) for phase, value in zip(PHASE_NAMES, phases.squeeze().numpy())}
    strongest = max(phase_scores, key=phase_scores.get)
    weakest = min(phase_scores, key=phase_scores.get)
    bowler_id = str(metadata.iloc[0]["bowler_id"])
    delivery_id = str(metadata.iloc[0]["delivery_id"])

    prediction = {
        "video_path": str(video_path),
        "bowler_id": bowler_id,
        "delivery_id": delivery_id,
        "overall_score": overall_score,
        "verdict": verdict_from_score(overall_score),
        "strongest_phase": strongest,
        "weakest_phase": weakest,
    }
    prediction.update({f"{phase}_score": score for phase, score in phase_scores.items()})

    pred_path = output_dir / OUTPUT_SUBDIRS["predictions"] / f"{delivery_id}_prediction.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([prediction]).to_csv(pred_path, index=False)

    dashboard_path = output_dir / OUTPUT_SUBDIRS["dashboard"] / f"{delivery_id}_dashboard.png"
    create_repeatability_dashboard(bowler_id, delivery_id, overall_score, phase_scores, dashboard_path)

    graph_path = output_dir / OUTPUT_SUBDIRS["graphs"] / f"{delivery_id}_lstm_sequence.png"
    plot_lstm_sequence_features(paths["sequence_csv"], graph_path)

    phase_video_path = analysis_root / f"{session_id}_7_phase_detection.mp4"
    render_phase_detection_video(
        Path(video_artifacts["analysis_video"]),
        paths["confirmed_phases"],
        phase_video_path,
    )

    print(f"[REPEATABILITY] Overall score: {overall_score:.1f}/100")
    print(f"[REPEATABILITY] Verdict: {prediction['verdict']}")
    print(f"[REPEATABILITY] Strongest phase: {strongest}")
    print(f"[REPEATABILITY] Weakest phase: {weakest}")
    print(f"[REPEATABILITY] Saved prediction: {pred_path}")
    print(f"[REPEATABILITY] Saved dashboard: {dashboard_path}")
    print(f"[REPEATABILITY] Saved 7-phase video: {phase_video_path}")
    save_manifest(video_path, output_dir, [
        analysis_root,
        paths["repeat_input"],
        paths["auto_phases"],
        paths["auto_events"],
        paths["confirmed_phases"],
        paths["confirmed_events"],
        paths["features"],
        paths["phase_labelled"],
        paths["curves"],
        paths["sequence_csv"],
        paths["sequence_npy"],
        paths["metadata_csv"],
        pred_path,
        dashboard_path,
        graph_path,
        phase_video_path,
    ])
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Score repeatability for one side-on video.")
    parser.add_argument("video_arg", nargs="?", help="Path to side-on video.")
    parser.add_argument("--video", default=None, help="Path to side-on video.")
    parser.add_argument("--video_dir", default=str(default_video_dir()), help="Folder to list when no video is provided.")
    parser.add_argument("--top_level_only", action="store_true", help="Only list videos directly inside --video_dir.")
    parser.add_argument("--model_path", default="outputs/repeatability/models/sideon_lstm.pt")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Deprecated; scoring always regenerates this video's outputs.")
    args = parser.parse_args()
    video = args.video or args.video_arg
    if not video:
        video_path = choose_video_interactive(resolve_path(args.video_dir), recursive=not args.top_level_only)
    else:
        video_path = resolve_path(video)

    score_video(
        video_path,
        resolve_path(args.model_path),
        resolve_path(args.output_dir),
        force=args.force,
    )


if __name__ == "__main__":
    main()

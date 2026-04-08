"""
run_single_video.py
--------------------
Interactive test runner for single video analysis.
Asks for video input and runs the pipeline.

USAGE:
    python run_single_video.py
    python run_single_video.py --video bowling_video.mp4
    python run_single_video.py --no-ball
"""

import os
import sys
import argparse
import yaml
import shutil

# Windows fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from main import run_pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


CONFIG = load_config()


def get_default_video_dirs():
    configured_dir = resolve_config_path(CONFIG["paths"]["input_single"])
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    fallback_input = os.path.join(workspace_root, "input_videos")
    fallback_single = os.path.join(fallback_input, "single")

    dirs = []
    for d in (configured_dir, fallback_single, fallback_input):
        if d not in dirs:
            dirs.append(d)
    return dirs


def list_available_videos():
    videos = []
    for input_dir in get_default_video_dirs():
        if not os.path.isdir(input_dir):
            continue
        for file in sorted(os.listdir(input_dir)):
            if file.lower().endswith(".mp4"):
                videos.append(os.path.join(input_dir, file))
        if videos:
            break

    configured_dir = resolve_config_path(CONFIG["paths"]["input_single"])
    if not videos and not os.path.isdir(configured_dir):
        os.makedirs(configured_dir, exist_ok=True)

    return videos


def interactive_video_selection():
    """Ask user to select a video file from the available input folders."""
    videos = list_available_videos()

    if not videos:
        print("[ERROR] No .mp4 files found in any of these folders:")
        for input_dir in get_default_video_dirs():
            print(f"  {input_dir}")
        print("[INFO] Please add videos to one of those folders or use --video <path>")
        return None

    current_dir = os.path.dirname(videos[0])
    print(f"\n[VIDEO] Available videos in {current_dir}:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")

    while True:
        try:
            choice = input("\nEnter video number (1-{}): ".format(len(videos))).strip()
            if not choice:
                print("[ERROR] Please enter a valid number")
                continue

            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                print(f"\n[OK] Selected: {videos[idx]}")
                return videos[idx]
            print(f"[ERROR] Invalid choice. Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")


def run_single_video_analysis(video_file: str, skip_ball: bool = False):
    """
    Run analysis on a single video file.

    Args:
        video_file : path to .mp4 file
        skip_ball  : if True, skip ball tracking
    """

    if not os.path.exists(video_file):
        print(f"[ERROR] Video not found: {video_file}")
        sys.exit(1)

    video_file = os.path.abspath(video_file)
    
    if os.path.isdir(video_file):
        session_path = video_file
        selected_video = None
        print(f"[VIDEO] Using video folder: {session_path}")
    else:
        if not video_file.lower().endswith(".mp4"):
            print(f"[ERROR] The path must point to an .mp4 video: {video_file}")
            sys.exit(1)
        
        # Create session folder in data/raw/sessions with video name
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        session_name = f"session_{video_name}"
        raw_sessions_dir = resolve_config_path(CONFIG["paths"]["raw_sessions"])
        session_path = os.path.join(raw_sessions_dir, session_name)
        os.makedirs(session_path, exist_ok=True)
        
        # Copy video to session folder
        dest_video = os.path.join(session_path, os.path.basename(video_file))
        shutil.copy(video_file, dest_video)
        
        selected_video = None
        print(f"[VIDEO] Found video: {video_file}")
        print(f"[INFO] Copied to session folder: {session_path}")

    output_dir = resolve_config_path(CONFIG["paths"]["outputs_single"])
    processed_dir = resolve_config_path(CONFIG["paths"]["processed_sessions"])

    try:
        print("\n" + "=" * 70)
        print("[START] STARTING PIPELINE (SINGLE VIDEO MODE)")
        print("=" * 70 + "\n")

        record = run_pipeline(
            session_path=session_path,
            run_ball=not skip_ball,
            output_dir=output_dir,
            single_video=True,
        )

        print("\n" + "=" * 70)
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n[RESULTS] Delivery Record:")
        for key, value in record.items():
            if value is not None:
                print(f"  {key}: {value}")

        print(f"\n[OUTPUT] Results saved to:")
        print(f"  Video: {output_dir}/{os.path.basename(session_path)}_bowling_video_analysis.avi")
        print(f"  CSV: {processed_dir}/{os.path.basename(session_path)}/results/deliveries.csv")

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[ERROR] PIPELINE FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Video Cricket Bowling Analysis")
    parser.add_argument("--video", default=None,
                        help="Path to video file (if not provided, will ask for input)")
    parser.add_argument("--no-ball", action="store_true",
                        help="Skip ball tracking (faster)")
    args = parser.parse_args()

    if args.video is None:
        video_file = interactive_video_selection()
        if video_file is None:
            sys.exit(1)
    else:
        video_file = args.video

    success = run_single_video_analysis(video_file, skip_ball=args.no_ball)
    sys.exit(0 if success else 1)

"""
run_four_camera.py
-------------------
Runner for 4-camera cricket bowling analysis.
Expects 4 videos: front.mp4, back.mp4, left.mp4, right.mp4

USAGE:
    python run_four_camera.py
    python run_four_camera.py --session session_001
    python run_four_camera.py --camera front
    python run_four_camera.py --no-ball
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


def check_four_camera_folder():
    """Check if input_videos/four_camera/ has all 4 required videos."""
    input_dir = resolve_config_path(CONFIG["paths"]["input_four_camera"])
    
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Folder not found: {input_dir}")
        return False, None
    
    required_views = ["front", "back", "left", "right"]
    missing = []
    
    for view in required_views:
        video_path = os.path.join(input_dir, f"{view}.mp4")
        if not os.path.exists(video_path):
            missing.append(f"{view}.mp4")
    
    if missing:
        print(f"[ERROR] Missing videos in {input_dir}:")
        for v in missing:
            print(f"  - {v}")
        print(f"\n[INFO] Required videos:")
        for view in required_views:
            print(f"  - {view}.mp4")
        return False, None
    
    print(f"[OK] All 4 camera videos found in {input_dir}")
    return True, input_dir


def run_four_camera_analysis(session_name: str, camera: str = "front", skip_ball: bool = False):
    """
    Run analysis on 4-camera session.
    
    Args:
        session_name : name for this session (e.g. "session_001")
        camera       : which camera to analyze (front, back, left, right)
        skip_ball    : if True, skip ball tracking
    """
    
    # Check if 4-camera folder has all videos
    valid, input_dir = check_four_camera_folder()
    if not valid:
        sys.exit(1)
    
    if not session_name.strip():
        print("[ERROR] Please provide a session name")
        sys.exit(1)
    
    session_name = session_name.strip().replace(" ", "_")
    
    # Create session folder in data/raw/sessions
    raw_sessions_dir = resolve_config_path(CONFIG["paths"]["raw_sessions"])
    session_path = os.path.join(raw_sessions_dir, session_name)
    os.makedirs(session_path, exist_ok=True)
    print(f"[INFO] Created session folder: {session_path}")
    
    # Copy all 4 videos to session folder
    try:
        for view in ["front", "back", "left", "right"]:
            src = os.path.join(input_dir, f"{view}.mp4")
            dest = os.path.join(session_path, f"{view}.mp4")
            shutil.copy(src, dest)
            print(f"[OK] Copied {view}.mp4")
    except Exception as e:
        print(f"[ERROR] Failed to copy videos: {str(e)}")
        sys.exit(1)
    
    output_dir = resolve_config_path(CONFIG["paths"]["outputs_four_camera"])
    processed_dir = resolve_config_path(CONFIG["paths"]["processed_sessions"])
    
    try:
        print("\n" + "=" * 70)
        print("[START] STARTING PIPELINE (4-CAMERA MODE)")
        print("=" * 70 + "\n")
        
        # Run pipeline in 4-camera mode
        record = run_pipeline(
            session_path=session_path,
            camera=camera,
            run_ball=not skip_ball,
            output_dir=output_dir,
            single_video=False,  # 4-camera mode
        )
        
        print("\n" + "=" * 70)
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n[RESULTS] Delivery Record:")
        for key, value in record.items():
            if value is not None:
                print(f"  {key}: {value}")
        
        print(f"\n[OUTPUT] Results saved to:")
        print(f"  Video: {output_dir}/{session_name}_{camera}_analysis.avi")
        print(f"  CSV: {processed_dir}/{session_name}/results/deliveries.csv")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[ERROR] PIPELINE FAILED: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up session folder
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            print(f"\n[CLEANUP] Cleaned up session folder: {session_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Camera Cricket Bowling Analysis")
    parser.add_argument("--session", default="session_001",
                        help="Session name (default: session_001)")
    parser.add_argument("--camera", default="front",
                        choices=["front", "back", "left", "right"],
                        help="Which camera to analyze (default: front)")
    parser.add_argument("--no-ball", action="store_true",
                        help="Skip ball tracking (faster)")
    args = parser.parse_args()
    
    success = run_four_camera_analysis(
        session_name=args.session,
        camera=args.camera,
        skip_ball=args.no_ball
    )
    sys.exit(0 if success else 1)

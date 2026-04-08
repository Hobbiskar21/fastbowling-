"""
src/ingestion/session_loader.py
---------------------------------
First module to run. Validates the session folder and returns
a SessionConfig dict used by every downstream module.
"""

import os
from src.utils.config_loader import get_config
from src.utils.video_utils import get_video_info


def load_session(session_path: str, single_video: bool = False, video_file: str = None) -> dict:
    """
    Validate a session folder and return a SessionConfig.
    Supports both 4-camera sessions and single-video mode.

    Args:
        session_path : path to folder containing video files
        single_video : if True, look for an .mp4 file in single-video mode
                      if False, expect front.mp4, back.mp4, left.mp4, right.mp4
        video_file  : optional explicit video path when single_video is True

    Returns:
        SessionConfig dict:
        {
            session_id, session_path, video_paths,
            fps, width, height, frame_count, single_video_mode
        }
    """
    cfg        = get_config()
    views      = cfg["camera"]["views"]
    session_id = os.path.basename(session_path.rstrip("/").rstrip("\\"))

    video_paths = {}

    if single_video:
        if video_file is not None:
            video_path = os.path.abspath(video_file)
            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Video not found: {video_path}"
                )
            view_name = os.path.splitext(os.path.basename(video_path))[0]
            video_paths[view_name] = video_path
            print(f"[OK] Single video mode: using {video_path} as '{view_name}'")
        else:
            # Single video mode: find any .mp4 file
            mp4_files = [f for f in os.listdir(session_path) if f.endswith(".mp4")]
            if not mp4_files:
                raise FileNotFoundError(
                    f"No .mp4 files found in {session_path}"
                )
            video_file = mp4_files[0]
            video_path = os.path.join(session_path, video_file)
            # Use the video file name (without extension) as the view name
            view_name = os.path.splitext(video_file)[0]
            video_paths[view_name] = video_path
            print(f"[OK] Single video mode: using {video_file} as '{view_name}'")
    else:
        # Multi-camera mode: expect all 4 cameras
        for view in views:
            path = os.path.join(session_path, f"{view}.mp4")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing camera file: {path}\n"
                    f"Expected {view}.mp4 inside {session_path}"
                )
            video_paths[view] = path

    # Read metadata from each camera
    infos = {view: get_video_info(path) for view, path in video_paths.items()}

    # Warn if FPS or resolution doesn't match across cameras (only for multi-camera)
    if not single_video and len(views) > 1:
        ref_view = views[0]
        ref_info = infos[ref_view]
        for view in views[1:]:
            info = infos[view]
            if abs(info["fps"] - ref_info["fps"]) > 1:
                print(f"[WARNING] FPS mismatch: {ref_view}={ref_info['fps']} "
                      f"vs {view}={info['fps']}")
            if info["width"] != ref_info["width"] or info["height"] != ref_info["height"]:
                print(f"[WARNING] Resolution mismatch: {ref_view} vs {view}")

    # Get reference info from first video
    ref_view = list(video_paths.keys())[0]
    ref_info = infos[ref_view]

    session = {
        "session_id":        session_id,
        "session_path":      session_path,
        "video_paths":       video_paths,
        "fps":               ref_info["fps"],
        "width":             ref_info["width"],
        "height":            ref_info["height"],
        "frame_count":       ref_info["frame_count"],
        "single_video_mode": single_video,
    }

    mode_str = "single video" if single_video else "4-camera"
    print(f"[OK] Session loaded ({mode_str}): {session_id} | "
          f"{ref_info['fps']}fps | "
          f"{ref_info['width']}x{ref_info['height']} | "
          f"{ref_info['frame_count']} frames")

    return session
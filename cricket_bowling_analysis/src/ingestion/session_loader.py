"""
src/ingestion/session_loader.py
# CAMERA: side | front | back

Loads a session folder with 3 separate camera subfolders.
Validates that each camera folder contains exactly one .mp4 file.
Returns a SessionConfig dict with metadata for all 3 cameras.

Input structure:
    input_videos/
    ├── side/
    │   └── side.mp4
    ├── front/
    │   └── front.mp4
    └── back/
        └── back.mp4
"""

import os
from typing import Optional, Dict
from src.utils.config_loader import get_config
from src.utils.video_utils import get_video_info


def load_session(session_path: str) -> Dict:
    """
    Load a session with 3 camera subfolders (side, front, back).
    
    Parameters
    ----------
    session_path : str
        Path to session folder containing side/, front/, back/ subfolders
    
    Returns
    -------
    dict
        SessionConfig with structure:
        {
            "session_id": str,
            "session_path": str,
            "cameras": {
                "side":  {"path": str, "fps": float, "width": int, "height": int, "frame_count": int},
                "front": {"path": str, "fps": float, "width": int, "height": int, "frame_count": int} or None,
                "back":  {"path": str, "fps": float, "width": int, "height": int, "frame_count": int} or None,
            }
        }
    """
    cfg = get_config()
    session_id = os.path.basename(session_path.rstrip("/").rstrip("\\"))
    
    print(f"\n[INGESTION] Loading session: {session_id}")
    print(f"[INGESTION] Session path: {session_path}")
    
    cameras = {}
    camera_names = ["side", "front", "back"]
    
    for camera_name in camera_names:
        camera_path = os.path.join(session_path, camera_name)
        
        if not os.path.exists(camera_path):
            print(f"[INGESTION] [WARNING] Camera folder not found: {camera_name}")
            cameras[camera_name] = None
            continue
        
        # Find .mp4 file in camera folder
        mp4_files = [f for f in os.listdir(camera_path) if f.endswith(".mp4")]
        
        if not mp4_files:
            print(f"[INGESTION] [WARNING] No .mp4 file found in {camera_name}/ folder")
            cameras[camera_name] = None
            continue
        
        if len(mp4_files) > 1:
            print(f"[INGESTION] [WARNING] Multiple .mp4 files in {camera_name}/ — using first: {mp4_files[0]}")
        
        video_path = os.path.join(camera_path, mp4_files[0])
        
        try:
            info = get_video_info(video_path)
            cameras[camera_name] = {
                "path": video_path,
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "frame_count": info["frame_count"],
            }
            print(f"[INGESTION] OK {camera_name}: {info['fps']}fps | {info['width']}x{info['height']} | {info['frame_count']} frames")
        except Exception as e:
            print(f"[INGESTION] [ERROR] Failed to load {camera_name}: {e}")
            cameras[camera_name] = None
    
    # Validate that at least side camera exists
    if cameras["side"] is None:
        raise FileNotFoundError(f"[INGESTION] Side camera is required but not found in {session_path}")
    
    # Warn if fps or resolution differs
    side_info = cameras["side"]
    for cam_name in ["front", "back"]:
        if cameras[cam_name] is not None:
            cam_info = cameras[cam_name]
            if abs(cam_info["fps"] - side_info["fps"]) > 1:
                print(f"[INGESTION] [WARNING] FPS mismatch: side={side_info['fps']} vs {cam_name}={cam_info['fps']}")
            if cam_info["width"] != side_info["width"] or cam_info["height"] != side_info["height"]:
                print(f"[INGESTION] [WARNING] Resolution mismatch: side={side_info['width']}x{side_info['height']} vs {cam_name}={cam_info['width']}x{cam_info['height']}")
    
    session_config = {
        "session_id": session_id,
        "session_path": session_path,
        "cameras": cameras,
    }
    
    print(f"[INGESTION] Session loaded successfully")
    return session_config

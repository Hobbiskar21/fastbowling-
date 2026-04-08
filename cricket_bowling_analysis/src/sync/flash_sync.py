"""
src/sync/flash_sync.py
------------------------
Detects the LED flash frame in each camera video.
Returns one frame index per camera used to align all 4 feeds.

HOW TO USE:
    Fire a bright LED visible to all 4 cameras at session start.
    Adjust flash_brightness_threshold in config.yaml if not detected.
"""

import cv2
from src.utils.config_loader import get_config


def detect_flash_frame(video_path: str, threshold: float = 200) -> int:
    """
    Scan video for the first frame where mean brightness > threshold.

    Returns:
        Frame index of flash. Returns 0 if not found.
    """
    cap       = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        brightness = frame.mean()
        if brightness > threshold:
            cap.release()
            print(f"[SYNC] Flash at frame {frame_idx} "
                  f"(brightness={brightness:.1f}) — {video_path}")
            return frame_idx
        frame_idx += 1

    cap.release()
    print(f"[WARNING] No flash detected in {video_path} — using frame 0")
    return 0


def detect_flash_all_cameras(video_paths: dict, single_video_mode: bool = False) -> dict:
    """
    Run flash detection on all cameras (1 or 4).
    For single video mode, returns flash frame for that video.
    For multi-camera mode, returns flash frames for all 4 cameras.

    Args:
        video_paths      : {view: path} from session
        single_video_mode : if True, skip flash detection (return 0)

    Returns:
        {view: int} — flash frame index per camera
    """
    cfg       = get_config()
    threshold = cfg["sync"]["flash_brightness_threshold"]
    
    if single_video_mode:
        # For single video, skip flash detection (no sync needed)
        print("[SYNC] Single video mode: skipping flash detection")
        return {view: 0 for view in video_paths.keys()}
    
    return {view: detect_flash_frame(path, threshold)
            for view, path in video_paths.items()}
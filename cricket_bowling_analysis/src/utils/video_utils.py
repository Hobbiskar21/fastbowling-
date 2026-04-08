"""
src/utils/video_utils.py
--------------------------
Shared OpenCV helpers used across the project.
- get_video_info() : read fps, width, height, frame count
- extract_frames() : read frames into a list
- write_video()    : write annotated frames to .mp4
- frame_to_rgb()   : BGR to RGB for MediaPipe
"""

import cv2
import numpy as np
from typing import Optional


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = {
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frames(path: str,
                   start_frame: int = 0,
                   end_frame: Optional[int] = None) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < start_frame:
            idx += 1
            continue
        if end_frame is not None and idx >= end_frame:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def write_video(frames: list, output_path: str, fps: float, slowdown: float = 1.0) -> None:
    """
    Write frames to video file.
    
    Args:
        frames: List of frame arrays
        output_path: Path to save video
        fps: Original video FPS
        slowdown: Slowdown factor (0.25 = 4x slower, 0.5 = 2x slower, 1.0 = normal)
    """
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    # Adjust FPS based on slowdown factor
    output_fps = fps * slowdown
    # Use XVID codec — most reliable on Windows
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_frame(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
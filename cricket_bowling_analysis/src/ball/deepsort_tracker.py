"""
src/ball/deepsort_tracker.py
------------------------------
Tracks the ball across frames using simple centroid tracking.
For single ball tracking, centroid matching is more reliable than DeepSORT.

Returns (cx, cy) per frame or None.
"""

import numpy as np
from src.utils.config_loader import get_config


def track_ball_sequence(detections: list, fps: float, frames: list = None) -> list:
    """
    Track ball using simple centroid matching.
    For single ball (cricket), this is more reliable than DeepSORT.

    Args:
        detections : list from yolo_detector — dict or None per frame.
        fps        : video frame rate.
        frames     : list of video frames (optional, unused).

    Returns:
        List of (cx, cy) tuples or None per frame.
    """
    cfg = get_config()["ball"]
    max_distance = cfg.get("tracker_max_distance", 100)  # pixels
    
    tracked_positions = []
    last_position = None
    
    for i, det in enumerate(detections):
        position = None
        
        if det is not None:
            current_pos = (det["cx"], det["cy"])
            
            # If we have a previous position, check if this detection is close
            if last_position is not None:
                distance = np.sqrt((current_pos[0] - last_position[0])**2 + 
                                 (current_pos[1] - last_position[1])**2)
                
                # If detection is close to last position, it's the same ball
                if distance < max_distance:
                    position = current_pos
                    last_position = current_pos
                else:
                    # Detection too far, might be noise - use prediction
                    position = last_position
            else:
                # First detection
                position = current_pos
                last_position = current_pos
        else:
            # No detection - use last known position (Kalman-like prediction)
            if last_position is not None:
                position = last_position
        
        tracked_positions.append(position)
    
    found = sum(1 for p in tracked_positions if p is not None)
    print(f"[TRACK] Ball tracked in {found}/{len(tracked_positions)} frames.")
    return tracked_positions
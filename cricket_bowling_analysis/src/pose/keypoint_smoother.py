"""
src/pose/keypoint_smoother.py
-------------------------------
Applies Savitzky-Golay smoothing to all 33 landmarks across the sequence.
Removes MediaPipe jitter while preserving peaks (needed for release detection).
Returns SmoothedLandmark objects with same .x .y .z .visibility interface.
"""

import numpy as np
from src.utils.math_utils import smooth_sequence
from src.utils.config_loader import get_config


class SmoothedLandmark:
    """Drop-in replacement for MediaPipe NormalizedLandmark."""
    def __init__(self, x, y, z, visibility):
        self.x          = x
        self.y          = y
        self.z          = z
        self.visibility = visibility


def smooth_landmarks_sequence(landmarks_sequence: list) -> list:
    """
    Smooth x and y of all 33 landmarks across all frames.

    Args:
        landmarks_sequence : list of MediaPipe landmark lists or None.

    Returns:
        List of SmoothedLandmark lists or None — same length as input.
    """
    cfg    = get_config()["pose"]
    window = cfg["smoothing_window"]
    poly   = cfg["smoothing_polyorder"]
    n      = len(landmarks_sequence)

    # Collect x, y per landmark across all frames
    xs = [[] for _ in range(33)]
    ys = [[] for _ in range(33)]

    for lm_list in landmarks_sequence:
        for j in range(33):
            if lm_list is not None:
                xs[j].append(lm_list[j].x)
                ys[j].append(lm_list[j].y)
            else:
                xs[j].append(None)
                ys[j].append(None)

    smoothed_xs = [smooth_sequence(xs[j], window, poly) for j in range(33)]
    smoothed_ys = [smooth_sequence(ys[j], window, poly) for j in range(33)]

    result = []
    for i in range(n):
        if landmarks_sequence[i] is None:
            result.append(None)
            continue

        frame_landmarks = []
        for j in range(33):
            original = landmarks_sequence[i][j]
            frame_landmarks.append(SmoothedLandmark(
                x=smoothed_xs[j][i] if smoothed_xs[j][i] is not None else original.x,
                y=smoothed_ys[j][i] if smoothed_ys[j][i] is not None else original.y,
                z=original.z,
                visibility=original.visibility,
            ))
        result.append(frame_landmarks)

    return result

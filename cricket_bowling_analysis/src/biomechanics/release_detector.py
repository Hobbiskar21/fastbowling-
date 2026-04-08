"""
src/biomechanics/release_detector.py
--------------------------------------
Finds the exact frame the ball is released.

Combines two signals:
    1. Wrist velocity peak (always used)
    2. Wrist-to-ball distance spike (used if ball tracking available)

Searches only within the DELIVERY phase window.
"""

import numpy as np
from typing import Optional


def find_release_frame(wrist_velocity_seq: list,
                       ball_positions: list = None,
                       wrist_positions: list = None,
                       phase_map: dict = None) -> Optional[int]:
    """
    Find the release frame index.

    Args:
        wrist_velocity_seq : list of float or None per frame.
        ball_positions     : list of (x, y) or None per frame (from DeepSORT).
        wrist_positions    : list of (x, y) or None per frame.
        phase_map          : dict frame_idx -> phase. Restricts search to DELIVERY.

    Returns:
        Frame index (int) or None if not detected.
    """
    n = len(wrist_velocity_seq)

    if phase_map is not None:
        search_frames = [i for i in range(n) if phase_map.get(i) == "DELIVERY"]
    else:
        search_frames = list(range(n))

    if not search_frames:
        return None

    # Signal 1: wrist velocity
    wrist_score = np.zeros(n)
    for i in search_frames:
        v = wrist_velocity_seq[i]
        if v is not None:
            wrist_score[i] = v

    # Signal 2: wrist-to-ball distance
    ball_score = np.zeros(n)
    if ball_positions is not None and wrist_positions is not None:
        for i in search_frames:
            bp = ball_positions[i]
            wp = wrist_positions[i]
            if bp is not None and wp is not None:
                ball_score[i] = np.linalg.norm(
                    np.array(bp) - np.array(wp)
                )

    # Normalize and combine
    def normalize(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    combined = normalize(wrist_score)
    if ball_positions is not None:
        combined += normalize(ball_score)

    # Mask to search window
    mask = np.zeros(n)
    for i in search_frames:
        mask[i] = 1
    combined *= mask

    release_frame = int(np.argmax(combined))
    return release_frame if combined[release_frame] > 0 else None
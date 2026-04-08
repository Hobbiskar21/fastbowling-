"""
src/ball/trajectory_fitter.py
-------------------------------
Fits a parabolic arc to post-release ball positions.
Estimates ball speed (pixels/sec) and release angle (degrees).
Convert to km/h once you know pixels-per-metre from calibration.
Wicket height = 71.1cm is a good reference object for scale.
"""

import numpy as np
from typing import Optional


def fit_ball_trajectory(tracked_positions: list,
                        release_frame: int,
                        fps: float,
                        window: int = 10) -> dict:
    """
    Estimate ball speed and release angle from post-release positions.

    Args:
        tracked_positions : list of (cx, cy) or None per frame.
        release_frame     : frame index of ball release.
        fps               : video frame rate.
        window            : post-release frames to use.

    Returns:
        {ball_speed_px_per_sec, release_angle_deg, fit_quality}
    """
    if release_frame is None:
        return {"ball_speed_px_per_sec": None,
                "release_angle_deg":     None,
                "fit_quality":           None}

    post_release = []
    for i in range(release_frame,
                   min(release_frame + window, len(tracked_positions))):
        pos = tracked_positions[i]
        if pos is not None:
            post_release.append((i - release_frame, pos[0], pos[1]))

    if len(post_release) < 4:
        return {"ball_speed_px_per_sec": None,
                "release_angle_deg":     None,
                "fit_quality":           None}

    xs = np.array([p[1] for p in post_release])
    ys = np.array([p[2] for p in post_release])

    # Release angle from x-y displacement
    dx            = xs[-1] - xs[0]
    dy            = ys[-1] - ys[0]
    release_angle = float(np.degrees(np.arctan2(-dy, dx)))

    # Speed from frame-to-frame displacement
    speeds = []
    for j in range(1, len(post_release)):
        d = np.sqrt((xs[j]-xs[j-1])**2 + (ys[j]-ys[j-1])**2)
        speeds.append(d * fps)
    ball_speed = float(np.mean(speeds)) if speeds else None

    # Parabola fit R² for quality
    try:
        coeffs = np.polyfit(xs, ys, 2)
        y_pred = np.polyval(coeffs, xs)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except Exception:
        r2 = None

    return {
        "ball_speed_px_per_sec": round(ball_speed, 2) if ball_speed else None,
        "release_angle_deg":     round(release_angle, 2),
        "fit_quality":           round(r2, 3) if r2 is not None else None,
    }
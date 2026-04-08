"""
src/visualization/metrics_overlay.py
---------------------------------------
Draws a semi-transparent HUD panel bottom-right of each frame.
Shows live angle and velocity values as the video plays.
Highlights in red if a value crosses a warning threshold:
    elbow > 165° → possible illegal action
    trunk_lean > 45° → injury risk
    lateral_flexion > 50° → excessive lateral bend
"""

import cv2
import numpy as np

THRESHOLDS = {
    "elbow_angle": 165,
    "trunk_lean":  45,
    "lateral_flexion": 50,
}


def draw_metrics_hud(frame: np.ndarray,
                     angles: dict,
                     velocities: dict = None,
                     ball_speed: float = None) -> np.ndarray:
    """
    Draw metrics HUD on a frame.

    Returns:
        Annotated BGR frame (copy).
    """
    out   = frame.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    h, w  = out.shape[:2]

    panel_w = 280
    panel_h = 300
    px      = w - panel_w - 10
    py      = h - panel_h - 10

    overlay = out.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    metrics = _build_metrics(angles, velocities, ball_speed)
    y = py + 20

    for label, value, warn in metrics:
        if value is None:
            text  = f"{label}: --"
            color = (100, 100, 100)
        else:
            text  = f"{label}: {value:.1f}"
            color = (0, 50, 255) if warn else (220, 220, 220)
        cv2.putText(out, text, (px + 10, y),
                    font, 0.5, color, 1, cv2.LINE_AA)
        y += 22

    return out


def _build_metrics(angles, velocities, ball_speed):
    metrics = []

    angle_display = [
        ("Elbow",        "elbow_angle"),
        ("Shoulder",     "shoulder_angle"),
        ("F.Knee",       "front_knee_angle"),
        ("B.Knee",       "back_knee_angle"),
        ("Hip",          "hip_angle"),
        ("Hip-Sh.Sep",   "hip_shoulder_sep"),
        ("Trunk Lean",   "trunk_lean"),
        ("Lat.Flexion",  "lateral_flexion"),
    ]

    for label, key in angle_display:
        val  = angles.get(key) if angles else None
        warn = val is not None and key in THRESHOLDS and val > THRESHOLDS[key]
        metrics.append((label, val, warn))

    if velocities:
        metrics.append(("Arm Vel.", velocities.get("arm_velocity_max"), False))
        metrics.append(("RunUp",    velocities.get("runup_speed_mean"),  False))

    if ball_speed is not None:
        metrics.append(("Ball Spd.", ball_speed, False))

    return metrics
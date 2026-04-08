"""
src/visualization/skeleton_drawer.py
--------------------------------------
Draws the MediaPipe skeleton on each frame.
Left side = green, right side = red, torso = cyan.
If the skeleton looks wrong here, your angles are wrong.
This is your primary visual debug tool.

Color convention — all values are BGR (OpenCV order):
    Cyan   (255, 255,   0)  — WRONG, that is yellow
    Cyan   (  0, 255, 255)  — correct
    Red    (  0,   0, 255)  — correct
    Green  (  0, 255,   0)  — correct
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Named landmark indices (MediaPipe Pose, 33-keypoint model)
# Using names avoids magic numbers throughout the module.
# ---------------------------------------------------------------------------
NOSE         = 0
L_SHOULDER   = 11
R_SHOULDER   = 12
L_ELBOW      = 13
R_ELBOW      = 14
L_WRIST      = 15
R_WRIST      = 16
L_HIP        = 23
R_HIP        = 24
L_KNEE       = 25
R_KNEE       = 26
L_ANKLE      = 27
R_ANKLE      = 28

# ---------------------------------------------------------------------------
# BGR color palette  ← all values are (Blue, Green, Red)
# ---------------------------------------------------------------------------
_CYAN       = (  0, 255, 255)   # correct cyan in BGR
_RED        = (  0,   0, 255)
_GREEN      = (  0, 255,   0)
_DARK_RED   = (  0,   0, 180)
_DARK_GREEN = (  0, 180,   0)
_GRAY       = (200, 200, 200)
_WHITE      = (255, 255, 255)
_YELLOW     = (  0, 255, 255)   # yellow in BGR
_MAGENTA    = (255,   0, 255)   # magenta in BGR
_ORANGE     = (  0, 165, 255)   # orange in BGR

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------
POSE_CONNECTIONS: list[tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER),   # torso
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP,      R_HIP),
    (R_SHOULDER, R_ELBOW),      # right arm
    (R_ELBOW,    R_WRIST),
    (L_SHOULDER, L_ELBOW),      # left arm
    (L_ELBOW,    L_WRIST),
    (R_HIP,      R_KNEE),       # right leg
    (R_KNEE,     R_ANKLE),
    (L_HIP,      L_KNEE),       # left leg
    (L_KNEE,     L_ANKLE),
    (NOSE,       L_SHOULDER),   # face → shoulders
    (NOSE,       R_SHOULDER),
]

CONNECTION_COLOR_MAP: dict[tuple[int, int], tuple[int, int, int]] = {
    (L_SHOULDER, R_SHOULDER): _CYAN,
    (L_SHOULDER, L_HIP):      _CYAN,
    (R_SHOULDER, R_HIP):      _CYAN,
    (L_HIP,      R_HIP):      _CYAN,
    (R_SHOULDER, R_ELBOW):    _RED,
    (R_ELBOW,    R_WRIST):    _RED,
    (L_SHOULDER, L_ELBOW):    _GREEN,
    (L_ELBOW,    L_WRIST):    _GREEN,
    (R_HIP,      R_KNEE):     _DARK_RED,
    (R_KNEE,     R_ANKLE):    _DARK_RED,
    (L_HIP,      L_KNEE):     _DARK_GREEN,
    (L_KNEE,     L_ANKLE):    _DARK_GREEN,
    (NOSE,       L_SHOULDER): _GRAY,
    (NOSE,       R_SHOULDER): _GRAY,
}

# HUD layout constants
_HUD_LINE_HEIGHT  = 22          # pixels between HUD text rows
_HUD_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT_SCALE   = 0.55
_HUD_FONT_THICKNESS = 1
_HUD_BG_ALPHA     = 0.55        # semi-transparent background


def _pixel(lm, width: int, height: int) -> tuple[int, int]:
    """Convert a normalised MediaPipe landmark to pixel coordinates."""
    return (int(lm.x * width), int(lm.y * height))


def _draw_hud(
    out: np.ndarray,
    angle_annotations: dict[str, float],
    phase_label: str | None,
    bowling_style: str | None = None,
    velocities: dict | None = None,
) -> None:
    """
    Render a semi-transparent HUD box in the bottom-left corner showing
    joint angles, phase label, bowling style, and velocities.

    Mutates *out* in-place.
    """
    if (not angle_annotations and phase_label is None and bowling_style is None
            and not velocities):
        return

    lines: list[str] = []
    if phase_label:
        lines.append(phase_label)
    if bowling_style:
        lines.append(f"Style: {bowling_style}")

    # Add angles in order
    angle_order = [
        ("Elbow", "elbow_angle"),
        ("Shoulder", "shoulder_angle"),
        ("F.Knee", "front_knee_angle"),
        ("B.Knee", "back_knee_angle"),
        ("Hip", "hip_angle"),
        ("Hip-Sh.Sep", "hip_shoulder_sep"),
        ("Trunk Lean", "trunk_lean"),
        ("Lat.Flexion", "lateral_flexion"),
    ]
    for label, key in angle_order:
        if key in angle_annotations:
            value = angle_annotations[key]
            # Only display if value is valid (not None and not NaN)
            if value is not None:
                try:
                    # Convert to float to ensure it's numeric
                    val_float = float(value)
                    if not np.isnan(val_float) and not np.isinf(val_float):
                        lines.append(f"{label}: {val_float:.1f}°")
                except (TypeError, ValueError):
                    # Skip if value can't be converted to float
                    pass

    # Add other angles not in the ordered list
    for name, value in angle_annotations.items():
        if value is not None and not any(key == name for _, key in angle_order):
            try:
                val_float = float(value)
                if not np.isnan(val_float) and not np.isinf(val_float):
                    lines.append(f"{name}: {val_float:.1f}°")
            except (TypeError, ValueError):
                # Skip if value can't be converted to float
                pass

    # Add velocities
    if velocities:
        arm_vel = velocities.get("arm_velocity_max")
        if arm_vel is not None:
            try:
                arm_vel_float = float(arm_vel)
                if not np.isnan(arm_vel_float) and not np.isinf(arm_vel_float):
                    lines.append(f"Arm Vel: {arm_vel_float:.1f}")
            except (TypeError, ValueError):
                pass
        
        runup_vel = velocities.get("runup_speed_mean")
        if runup_vel is not None:
            try:
                runup_vel_float = float(runup_vel)
                if not np.isnan(runup_vel_float) and not np.isinf(runup_vel_float):
                    lines.append(f"RunUp: {runup_vel_float:.1f}")
            except (TypeError, ValueError):
                pass

    if not lines:
        return

    padding   = 8
    line_h    = _HUD_LINE_HEIGHT
    box_w     = 200
    box_h     = padding * 2 + len(lines) * line_h
    frame_h, frame_w = out.shape[:2]
    x0 = padding
    y0 = frame_h - box_h - padding

    # Semi-transparent dark background
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, _HUD_BG_ALPHA, out, 1 - _HUD_BG_ALPHA, 0, out)

    for i, line in enumerate(lines):
        color = (0, 220, 255) if i == 0 and phase_label else _WHITE
        ty = y0 + padding + (i + 1) * line_h - 4
        # Ensure line is valid string before rendering
        if line and isinstance(line, str):
            cv2.putText(out, line, (x0 + padding, ty),
                        _HUD_FONT, _HUD_FONT_SCALE, color,
                        _HUD_FONT_THICKNESS, cv2.LINE_AA)


def draw_skeleton(
    frame: np.ndarray,
    landmarks,
    width: int,
    height: int,
    min_visibility: float = 0.5,
    line_thickness: int = 3,
    circle_radius: int = 6,
    angle_annotations: dict[str, float] | None = None,
    phase_label: str | None = None,
    bowling_style: str | None = None,
    velocities: dict | None = None,
) -> np.ndarray:
    """
    Draw the MediaPipe skeleton on a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame (not modified — a copy is returned).
    landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Per-frame landmark list from MediaPipe Pose.
    width, height : int
        Frame dimensions in pixels.
    min_visibility : float
        Landmarks below this visibility score are skipped (0–1).
    line_thickness : int
        Stroke width for skeleton lines.
    circle_radius : int
        Radius of joint-dot circles.
    angle_annotations : dict[str, float], optional
        Joint-angle values to display in the HUD.
    phase_label : str, optional
        Current phase name to show in the HUD.
    bowling_style : str, optional
        Bowling style classification to show in the HUD.
    velocities : dict, optional
        Velocity metrics (arm_velocity_max, runup_speed_mean) to display.

    Returns
    -------
    np.ndarray
        Annotated BGR frame (copy — original is not modified).
    """
    out = frame.copy()
    if landmarks is None:
        return out

    # Build a dict of visible pixel-space keypoints.
    points: dict[int, tuple[int, int]] = {}
    for i, lm in enumerate(landmarks):
        if lm.visibility >= min_visibility:
            points[i] = _pixel(lm, width, height)

    # Draw limb connections.
    for (a, b) in POSE_CONNECTIONS:
        if a in points and b in points:
            color = CONNECTION_COLOR_MAP.get((a, b), _WHITE)
            cv2.line(out, points[a], points[b], color,
                     line_thickness, cv2.LINE_AA)

    # Draw joint dots — radius 3 with bright detectable colors.
    for idx, pt in points.items():
        if idx == NOSE:
            cv2.circle(out, pt, 3, _YELLOW, -1, cv2.LINE_AA)
        elif idx in [L_SHOULDER, R_SHOULDER]:
            cv2.circle(out, pt, 3, _CYAN, -1, cv2.LINE_AA)
        elif idx in [L_ELBOW, R_ELBOW]:
            cv2.circle(out, pt, 3, _MAGENTA, -1, cv2.LINE_AA)
        elif idx in [L_WRIST, R_WRIST]:
            cv2.circle(out, pt, 3, _ORANGE, -1, cv2.LINE_AA)
        elif idx in [L_HIP, R_HIP]:
            cv2.circle(out, pt, 3, _GREEN, -1, cv2.LINE_AA)
        elif idx in [L_KNEE, R_KNEE]:
            cv2.circle(out, pt, 3, _RED, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, pt, 3, _WHITE, -1, cv2.LINE_AA)

    # Render HUD (angles + phase label + bowling style + velocities).
    _draw_hud(out, angle_annotations or {}, phase_label, bowling_style,
              velocities)

    return out

"""
src/biomechanics/angle_calculator.py
--------------------------------------
Computes joint angles from MediaPipe keypoints per frame.

Angles computed (all in degrees, 0-180):
    elbow_angle        : bowling arm elbow extension
    shoulder_angle     : bowling arm raised relative to body
    front_knee_angle   : front leg brace at delivery
    back_knee_angle    : back leg drive
    hip_angle          : hip position
    trunk_lean         : lateral tilt of torso
    hip_shoulder_sep   : angle between shoulder line and hip line
    lateral_flexion    : lateral bend of torso at release (nose-hip-shoulder)

To add a new angle:
    1. Extract the 3 joint landmarks
    2. Call calculate_angle(a, b, c)
    3. Add result to angles dict
"""

import numpy as np
from typing import Optional


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at joint B formed by A → B → C."""
    ba     = a - b
    bc     = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def get_landmark(landmarks: list, index: int, width: int, height: int,
                 min_confidence: float = 0.5) -> Optional[np.ndarray]:
    """Extract one landmark as pixel-space numpy array. Returns None if low confidence."""
    lm = landmarks[index]
    if lm.visibility < min_confidence:
        return None
    return np.array([lm.x * width, lm.y * height])


def compute_all_angles(landmarks: list, width: int, height: int,
                       min_confidence: float = 0.5) -> dict:
    """
    Compute all biomechanical angles for a single frame.

    Returns:
        dict of angle_name -> float (degrees) or None if landmarks missing.
    """
    def get(idx):
        return get_landmark(landmarks, idx, width, height, min_confidence)

    nose       = get(0)
    l_shoulder = get(11)
    r_shoulder = get(12)
    l_elbow    = get(13)
    r_elbow    = get(14)
    l_wrist    = get(15)
    r_wrist    = get(16)
    l_hip      = get(23)
    r_hip      = get(24)
    l_knee     = get(25)
    r_knee     = get(26)
    l_ankle    = get(27)
    r_ankle    = get(28)

    mid_hip = None
    if l_hip is not None and r_hip is not None:
        mid_hip = (l_hip + r_hip) / 2.0

    angles = {}

    # Bowling arm elbow (right arm)
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["elbow_angle"] = calculate_angle(r_shoulder, r_elbow, r_wrist)
    else:
        angles["elbow_angle"] = None

    # Shoulder angle
    if nose is not None and r_shoulder is not None and r_elbow is not None:
        angles["shoulder_angle"] = calculate_angle(nose, r_shoulder, r_elbow)
    else:
        angles["shoulder_angle"] = None

    # Front knee (left knee for right-arm bowler)
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["front_knee_angle"] = calculate_angle(l_hip, l_knee, l_ankle)
    else:
        angles["front_knee_angle"] = None

    # Back knee (right knee for right-arm bowler)
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["back_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)
    else:
        angles["back_knee_angle"] = None

    # Hip angle
    if nose is not None and r_hip is not None and r_knee is not None:
        angles["hip_angle"] = calculate_angle(nose, r_hip, r_knee)
    else:
        angles["hip_angle"] = None

    # Trunk lean
    if nose is not None and mid_hip is not None and r_hip is not None:
        angles["trunk_lean"] = calculate_angle(nose, mid_hip, r_hip)
    else:
        angles["trunk_lean"] = None

    # Hip-shoulder separation
    if (l_shoulder is not None and r_shoulder is not None and
            l_hip is not None and r_hip is not None):
        shoulder_line = r_shoulder - l_shoulder
        hip_line      = r_hip - l_hip
        cos_sep = np.dot(shoulder_line, hip_line) / (
            np.linalg.norm(shoulder_line) * np.linalg.norm(hip_line) + 1e-8
        )
        angles["hip_shoulder_sep"] = float(
            np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
        )
    else:
        angles["hip_shoulder_sep"] = None

    # Lateral flexion angle (lateral bend at release)
    if nose is not None and mid_hip is not None and r_shoulder is not None:
        angles["lateral_flexion"] = calculate_angle(nose, mid_hip, r_shoulder)
    else:
        angles["lateral_flexion"] = None

    return angles


def summarize_angle_sequence(angle_sequence: list) -> dict:
    """
    Compute min, max, mean for each angle across all frames.

    Returns:
        {angle_name: {"min": float, "max": float, "mean": float}}
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for frame_angles in angle_sequence:
        for name, val in frame_angles.items():
            if val is not None:
                buckets[name].append(val)

    summary = {}
    for name, values in buckets.items():
        arr = np.array(values)
        summary[name] = {
            "min":  round(float(arr.min()), 2),
            "max":  round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
        }
    return summary
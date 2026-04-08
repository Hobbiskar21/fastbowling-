"""
src/pose/mediapipe_detector.py
--------------------------------
Runs MediaPipe Pose on every frame.
Returns 33 keypoints per frame or None if no person detected.

Landmark index reference (most used):
    0=nose
    11=L_shoulder  12=R_shoulder
    13=L_elbow     14=R_elbow
    15=L_wrist     16=R_wrist
    23=L_hip       24=R_hip
    25=L_knee      26=R_knee
    27=L_ankle     28=R_ankle
"""

import mediapipe as mp

# Try to use old API (mp.solutions), fall back to stub if not available
try:
    mp_pose = mp.solutions.pose
    Pose = mp_pose.Pose
except AttributeError:
    # Newer mediapipe version without old API
    Pose = None

from src.utils.config_loader import get_config


def detect_pose_sequence(frames: list, width: int, height: int) -> list:
    """
    Run MediaPipe Pose on a list of BGR frames.

    Returns:
        List of landmark lists (33 landmarks) or None per frame.
    """
    if Pose is None:
        print("[POSE] MediaPipe Pose not available, returning None for all frames")
        return [None] * len(frames)
    
    cfg     = get_config()["pose"]
    pose    = Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    )

    landmarks_sequence = []

    for i, frame in enumerate(frames):
        rgb    = frame[:, :, ::-1]          # BGR to RGB
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks_sequence.append(result.pose_landmarks.landmark)
        else:
            landmarks_sequence.append(None)

        if i % 100 == 0:
            detected = sum(1 for l in landmarks_sequence if l is not None)
            print(f"[POSE] Frame {i}/{len(frames)} — "
                  f"detected in {detected}/{i+1}")

    pose.close()
    detected = sum(1 for l in landmarks_sequence if l is not None)
    print(f"[POSE] Done. Detected in {detected}/{len(frames)} frames.")
    return landmarks_sequence
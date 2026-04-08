"""
main.py
---------
CLI entry point. Runs the full pipeline on a session folder.
Supports both 4-camera sessions and single-video mode.

USAGE (4-camera mode):
    python main.py --session data/raw/sessions/session_001
    python main.py --session data/raw/sessions/session_001 --camera front
    python main.py --session data/raw/sessions/session_001 --no-ball

USAGE (single-video mode):
    python main.py --session data/raw/sessions/session_001 --single-video
    python main.py --session data/raw/sessions/session_001 --single-video --no-ball

OPTIONS:
    --session      : path to session folder (required)
    --camera       : which camera to run pose analysis on (default: front, 4-camera mode only)
    --no-ball      : skip YOLO + DeepSORT (faster, only pose angles)
    --output       : where to save annotated video (default: outputs)
    --single-video : enable single-video mode (auto-detect video file)
"""

import os
import argparse
import yaml
import numpy as np

# Windows fix — must be before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.ingestion.session_loader      import load_session
from src.ingestion.frame_extractor     import extract_all_cameras
from src.sync.flash_sync               import detect_flash_all_cameras
from src.sync.frame_aligner            import validate_alignment
from src.pose.mediapipe_detector       import detect_pose_sequence
from src.pose.keypoint_smoother        import smooth_landmarks_sequence
from src.biomechanics.angle_calculator   import compute_all_angles, summarize_angle_sequence
from src.biomechanics.velocity_estimator import compute_all_velocities
from src.biomechanics.phase_segmenter    import segment_phases
from src.biomechanics.release_detector   import find_release_frame
from src.biomechanics.feature_aggregator import build_delivery_record, validate_record
from src.biomechanics.bowling_style_detector import validate_camera_angle, classify_bowling_style, get_camera_angle_type
from src.biomechanics.runup_analyser import analyse_runup
from src.storage.csv_writer              import save_delivery
from src.visualization.skeleton_drawer   import draw_skeleton
from src.utils.video_utils               import write_video
from src.utils.config_loader             import get_config

# -- Load config ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()


def run_pipeline(session_path: str,
                 camera: str = "front",
                 run_ball: bool = True,
                 output_dir: str = "outputs",
                 single_video: bool = False,
                 video_file: str = None) -> dict:
    """
    Full pipeline: load → sync → pose → ball → biomechanics → save → render.
    Supports both 4-camera and single-video modes.

    Args:
        session_path : path to session folder
        camera       : which camera to analyze (for multi-camera mode)
        run_ball     : whether to run ball tracking
        output_dir   : where to save output video
        single_video : if True, treat as single video mode
        video_file   : optional explicit video path when single_video is True

    Returns:
        DeliveryRecord dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_config()

    print("\n-- Step 1: Load session --------------------------------------")
    session = load_session(session_path, single_video=single_video, video_file=video_file)

    print("\n-- Step 2: Sync cameras --------------------------------------")
    offsets   = detect_flash_all_cameras(session["video_paths"], 
                                         single_video_mode=single_video)
    alignment = validate_alignment(offsets)

    print("\n-- Step 3: Extract frames ------------------------------------")
    all_frames = extract_all_cameras(session, offsets)
    
    # For single video mode, use the available video name; for multi-camera, use specified camera
    if single_video:
        camera = list(all_frames.keys())[0]
        print(f"Single video mode: using '{camera}' camera")
    
    frames     = all_frames[camera]
    print(f"Using camera: {camera} | {len(frames)} frames")

    print("\n-- Step 4: Detect pose ---------------------------------------")
    w, h = session["width"], session["height"]
    fps  = session["fps"]
    landmarks_raw      = detect_pose_sequence(frames, w, h)
    landmarks_smoothed = smooth_landmarks_sequence(landmarks_raw)

    print("\n-- Step 4b: Validate camera angle ----------------------------")
    is_valid, camera_msg = validate_camera_angle(landmarks_smoothed, w, h)
    print(f"[INFO] {camera_msg}")

    # Detect camera angle type for feature selection
    camera_angle_type = get_camera_angle_type(landmarks_smoothed, w, h)
    print(f"[INFO] Detected camera angle: {camera_angle_type}")

    if camera_angle_type == 'side_on':
        print("[WARNING] Side-on angle detected")

    print("\n-- Step 5: Biomechanics (Phase Segmentation) ----------------")
    velocities = compute_all_velocities(landmarks_smoothed, fps, w, h)

    wrist_positions    = _extract_positions(landmarks_smoothed, 16, w, h)
    shoulder_positions = _extract_positions(landmarks_smoothed, 12, w, h)
    
    # Extract knee angles for phase segmentation
    knee_angles = []
    for lm in landmarks_smoothed:
        if lm is not None:
            # Compute knee angle (joints 25, 23, 24 for right leg)
            from src.biomechanics.angle_calculator import calculate_angle
            import numpy as np
            r_hip = np.array([lm[24].x * w, lm[24].y * h]) if lm[24].visibility > 0.5 else None
            r_knee = np.array([lm[26].x * w, lm[26].y * h]) if lm[26].visibility > 0.5 else None
            r_ankle = np.array([lm[28].x * w, lm[28].y * h]) if lm[28].visibility > 0.5 else None
            
            if r_hip is not None and r_knee is not None and r_ankle is not None:
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            else:
                knee_angle = 0
            knee_angles.append(knee_angle)
        else:
            knee_angles.append(0)

    # Call new segment_phases with updated signature
    phase_labels, phase_boundaries = segment_phases(
        wrist_positions,
        velocities["wrist_velocity_seq"],
        velocities["hip_velocity_seq"],
        knee_angles,
        fps
    )
    
    # Convert labels array to phase_map dict for compatibility
    phase_map = {i: phase_labels[i] for i in range(len(phase_labels))}
    p_ranges = phase_boundaries

    print("\n-- Step 5b: Analyze run-up ----------------------------------")
    runup_output_path = os.path.join(output_dir, f"{session['session_id']}_runup_analysis.png")
    runup_metrics = analyse_runup(
        landmarks_smoothed,
        phase_map,
        fps,
        w,
        h,
        runup_output_path,
        velocity_drop_threshold=0.3,
        gate_warn_multiplier=1.5,
        camera_angle_type=camera_angle_type,
    )
    print(f"[RUNUP] Analysis complete. Visualization saved to: {runup_output_path}")
    print(f"[RUNUP] Approach angle: {runup_metrics.get('approach_angle_deg', 0):.1f}°")
    print(f"[RUNUP] Stride count: {runup_metrics.get('stride_count', 0)}")

    release = find_release_frame(
        velocities["wrist_velocity_seq"],
        wrist_positions=wrist_positions,
        phase_map=phase_map,
    )
    print(f"Release frame: {release}")

    # Classify bowling style at release frame
    if release is not None and release < len(landmarks_smoothed):
        style, style_breakdown = classify_bowling_style(landmarks_smoothed[release], w, h)
        print(f"[STYLE] Bowling style: {style}")
        print(f"[STYLE] Front score: {style_breakdown['front_score']:.2f}, Side score: {style_breakdown['side_score']:.2f}")
        if style_breakdown['signals']:
            print(f"[STYLE] Signals: {style_breakdown['signals']}")
    else:
        style = "UNKNOWN"
        style_breakdown = {'style': 'UNKNOWN', 'front_score': 0.0, 'side_score': 0.0, 'signals': {}}

    angle_seq = [
        compute_all_angles(lm, w, h) if lm is not None else {}
        for lm in landmarks_smoothed
    ]
    angle_summary     = summarize_angle_sequence(angle_seq)
    angles_at_release = angle_seq[release] if release is not None else {}

    print("\n-- Step 6: Build delivery record ----------------------------")
    delivery_number = 1
    record = build_delivery_record(
        session_id        = session["session_id"],
        delivery_number   = delivery_number,
        bowler_name       = str(delivery_number),
        fps               = fps,
        phase_map         = phase_map,
        phase_ranges      = p_ranges,
        release_frame     = release,
        angles_at_release = angles_at_release,
        angle_summary     = angle_summary,
        velocities        = velocities,
        bowling_style     = style,
        bowling_style_breakdown = style_breakdown,
        runup_metrics     = runup_metrics,
    )

    warnings = validate_record(record)
    for msg in warnings:
        print(f"[WARNING] {msg}")

    print("\n-- Step 7: Save ----------------------------------------------")
    save_delivery(record, session["session_id"], single_video_mode=single_video)

    print("\n-- Step 8: Render annotated video ----------------------------")
    out_path = os.path.join(
        output_dir,
        f"{session['session_id']}_{camera}_analysis.avi"
    )
    
    # Check if output file already exists
    if os.path.exists(out_path):
        print(f"\n[WARNING] Output video already exists: {out_path}")
        response = input("Overwrite existing video? (yes/no): ").strip().lower()
        if response != "yes":
            print(f"[INFO] Skipping video rendering. Keeping existing file.")
            print(f"\n[DONE] Output saved to: {out_path}")
            return record
    
    annotated    = []

    for i, frame in enumerate(frames):
        lm     = landmarks_smoothed[i]
        phase  = phase_map.get(i, "")
        angles = angle_seq[i]

        # Use bowling style from release frame for all frames
        frame_style = style if i == release else None

        # Prepare all angle annotations for HUD
        angle_annotations = {}
        if angles:
            for key, value in angles.items():
                # Only include valid angles (not None, not NaN, not Inf)
                if value is not None:
                    try:
                        val_float = float(value)
                        if not np.isnan(val_float) and not np.isinf(val_float):
                            angle_annotations[key] = val_float
                    except (TypeError, ValueError):
                        # Skip if value can't be converted to float
                        pass

        # Prepare velocity data for HUD
        frame_velocities = {}
        if velocities:
            if "wrist_velocity_seq" in velocities and i < len(velocities["wrist_velocity_seq"]):
                wrist_vel = velocities["wrist_velocity_seq"][i]
                if wrist_vel is not None:
                    frame_velocities["arm_velocity_max"] = wrist_vel
            if "hip_velocity_seq" in velocities and i < len(velocities["hip_velocity_seq"]):
                hip_vel = velocities["hip_velocity_seq"][i]
                if hip_vel is not None:
                    frame_velocities["runup_speed_mean"] = hip_vel

        # Draw skeleton with HUD showing all metrics
        out = draw_skeleton(
            frame, lm, w, h,
            angle_annotations=angle_annotations,
            phase_label=phase,
            bowling_style=frame_style,
            velocities=frame_velocities if frame_velocities else None,
        )

        annotated.append(out)

    out_path = os.path.join(
        output_dir,
        f"{session['session_id']}_{camera}_analysis.avi"
    )
    write_video(annotated, out_path, fps, slowdown=0.25)
    print(f"\n[DONE] Output saved to: {out_path}")
    return record


def _extract_positions(landmarks_seq, joint_idx, width, height,
                        min_vis=0.5) -> list:
    positions = []
    for lm in landmarks_seq:
        if lm is not None and lm[joint_idx].visibility >= min_vis:
            positions.append((lm[joint_idx].x * width,
                               lm[joint_idx].y * height))
        else:
            positions.append(None)
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket Bowling Analysis")
    parser.add_argument("--session", 
                        default=CONFIG["paths"]["input_four_camera"],
                        help=f"Path to session folder (default: {CONFIG['paths']['input_four_camera']})")
    parser.add_argument("--camera",  default="front",
                        choices=["front", "back", "left", "right"])
    parser.add_argument("--no-ball", action="store_true",
                        help="Skip ball tracking")
    parser.add_argument("--output",  default=CONFIG["paths"]["outputs"])
    parser.add_argument("--single-video", action="store_true",
                        help="Single video mode (auto-detect video file)")
    args = parser.parse_args()

    run_pipeline(
        session_path = args.session,
        camera       = args.camera,
        run_ball     = not args.no_ball,
        output_dir   = args.output,
        single_video = args.single_video,
    )


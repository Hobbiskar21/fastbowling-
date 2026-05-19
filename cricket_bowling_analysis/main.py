"""
main.py
---------
CLI entry point. Runs the full pipeline on a session folder.
Uses MANUAL frame-based synchronization.
Normally run this through RUN_CLEAN.py.

USAGE (multi-camera mode with interactive sync):
    python RUN_CLEAN.py --session data/raw/sessions/session_001
    (will ask for frame numbers for sync event)

USAGE (multi-camera mode with saved sync offsets):
    python RUN_CLEAN.py --session data/raw/sessions/session_001 --sync-file outputs/session_001_sync_offsets.txt

OPTIONS:
    --session      : path to session folder (required)
    --camera       : which camera to analyze (default: side)
    --no-ball      : skip ball tracking (faster, only pose angles)
    --output       : where to save annotated video (default: outputs)
    --sync-file    : optional path to pre-saved sync offsets file
"""

import sys
import os

# FORCE PYTHON TO NOT USE BYTECODE - ALWAYS USE FRESH SOURCE CODE
sys.dont_write_bytecode = True

# Clear all __pycache__ before importing anything
import shutil
for root, dirs, files in os.walk("."):
    if "__pycache__" in dirs:
        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)

import argparse
import yaml
import numpy as np

# Windows fix — must be before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.ingestion.session_loader      import load_session
from src.ingestion.frame_extractor     import extract_all_cameras
from src.sync.manual_sync              import manual_sync_interactive, manual_sync_from_file, save_sync_offsets
from src.pose                           import detect_pose_sequence as detect_pose_yolo, smooth_coco_keypoints_sequence, SmoothedLandmark
from src.biomechanics.angle_calculator   import compute_all_angles, summarize_angle_sequence
from src.biomechanics.velocity_estimator import compute_all_velocities
from src.biomechanics.phase_segmenter    import segment_phases
from src.biomechanics.release_detector   import find_release_frame
from src.biomechanics.feature_aggregator import build_delivery_record, validate_record
from src.biomechanics.view_selector import ViewSelector
from src.biomechanics.bowling_style_detector import validate_camera_angle, classify_bowling_style, get_camera_angle_type
from src.biomechanics.runup_analyser import analyse_runup
from src.storage.csv_writer              import save_delivery
from src.visualization.skeleton_drawer   import draw_skeleton
from src.utils.video_utils               import write_video
from src.utils.config_loader             import get_config


def _draw_wrist_trail(frame, kpts_sequence, frame_idx, wrist_idx=10, history=10):
    """Draw a script-style wrist trail with a yellow current marker."""
    points = []
    start = max(0, frame_idx - history + 1)
    for idx in range(start, frame_idx + 1):
        if idx >= len(kpts_sequence):
            continue
        kpts = kpts_sequence[idx]
        if kpts is None or wrist_idx >= len(kpts):
            continue
        x, y = float(kpts[wrist_idx][0]), float(kpts[wrist_idx][1])
        if np.isnan(x) or np.isnan(y):
            continue
        points.append((int(x), int(y)))

    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)

    if points:
        cv2.circle(frame, points[-1], 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, points[-1], 2, (255, 255, 255), -1, cv2.LINE_AA)

    return frame

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
                 sync_file: str = None) -> dict:
    """
    Full pipeline: load → manual sync → pose → biomechanics → save → render.

    Args:
        session_path : path to session folder
        camera       : which camera to analyze
        run_ball     : whether to run ball tracking
        output_dir   : where to save output video
        sync_file    : optional path to pre-saved sync offsets file

    Returns:
        DeliveryRecord dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = get_config()

    print("\n-- Step 1: Load session --------------------------------------")
    session = load_session(session_path)

    print("\n-- Step 2: Manual synchronization ----------------------------")
    # Get offsets either from file or interactive input
    if sync_file and os.path.exists(sync_file):
        print(f"[SYNC] Loading offsets from file: {sync_file}")
        offsets = manual_sync_from_file(sync_file)
        if offsets is None:
            print("[ERROR] Failed to load sync file, falling back to interactive mode")
            offsets = manual_sync_interactive()
    else:
        print("[SYNC] Starting interactive manual synchronization...")
        offsets = manual_sync_interactive()
    
    # Save offsets for future use
    sync_output_file = os.path.join(output_dir, f"{session['session_id']}_sync_offsets.txt")
    save_sync_offsets(offsets, sync_output_file)

    print("\n-- Step 3: Extract frames ------------------------------------")
    all_frames = extract_all_cameras(session, offsets)
    
    camera_frames = all_frames[camera]
    frames = camera_frames["original"] if isinstance(camera_frames, dict) else camera_frames
    print(f"Using camera: {camera} | {len(frames)} frames")

    print("\n-- Step 4: Detect pose ---------------------------------------")
    w, h = session["width"], session["height"]
    fps  = session["fps"]
    
    # Detect pose with YOLO
    coco_landmarks_raw, pose_metadata = detect_pose_yolo(
        frames, w, h,
        pose_model="yolov8m-pose.pt",
        imgsz=960,
        max_jump_px=260.0,
        device="cuda"  # Use GPU for faster inference
    )
    
    # Smooth raw COCO keypoints in pixel space, matching the script logic.
    coco_landmarks_smoothed = smooth_coco_keypoints_sequence(coco_landmarks_raw)

    # Keep COCO format (17 keypoints) - convert to landmark objects for compatibility
    landmarks_raw = []
    for coco_kpts in coco_landmarks_smoothed:
        if coco_kpts is not None:
            # Convert COCO (17, 2) to list of SmoothedLandmark objects
            # YOLO returns pixel coordinates, need to normalize to 0-1
            landmarks = []
            for i in range(17):
                x_px, y_px = coco_kpts[i]
                # Normalize to 0-1 range
                x_norm = x_px / w if w > 0 else 0
                y_norm = y_px / h if h > 0 else 0
                landmarks.append(SmoothedLandmark(x_norm, y_norm, 0.0, 1.0))
            landmarks_raw.append(landmarks)
        else:
            landmarks_raw.append(None)

    landmarks_smoothed = landmarks_raw
    print("[INFO] Person tracking enabled - YOLO Pose detection")
    print(f"[INFO] Pose metadata: {pose_metadata}")

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
            from src.biomechanics.angle_calculator import calculate_angle
            import numpy as np
            r_hip = np.array([lm[12].x * w, lm[12].y * h]) if lm[12].visibility > 0.5 else None
            r_knee = np.array([lm[14].x * w, lm[14].y * h]) if lm[14].visibility > 0.5 else None
            r_ankle = np.array([lm[16].x * w, lm[16].y * h]) if lm[16].visibility > 0.5 else None
            
            if r_hip is not None and r_knee is not None and r_ankle is not None:
                knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            else:
                knee_angle = 0
            knee_angles.append(knee_angle)
        else:
            knee_angles.append(0)

    # Extract hip positions for phase segmentation
    hip_positions = []
    for lm in landmarks_smoothed:
        if lm is not None and lm[12].visibility > 0.5:  # Right hip (COCO index 12)
            hip_positions.append((lm[12].x * w, lm[12].y * h))
        else:
            hip_positions.append(None)
    
    # Extract ankle positions for jump detection (both feet off ground)
    ankle_positions = []
    for lm in landmarks_smoothed:
        if lm is not None:
            left_ankle = (lm[15].x * w, lm[15].y * h) if lm[15].visibility > 0.3 else None  # COCO index 15
            right_ankle = (lm[16].x * w, lm[16].y * h) if lm[16].visibility > 0.3 else None  # COCO index 16
            ankle_positions.append((left_ankle, right_ankle))
        else:
            ankle_positions.append((None, None))
    
    # Extract right elbow angles for delivery detection (arm straightening)
    # COCO indices: 6=rsho, 8=relb, 10=rwri
    elbow_angles = []
    for lm in landmarks_smoothed:
        if lm is not None:
            from src.biomechanics.angle_calculator import calculate_angle
            r_shoulder = np.array([lm[6].x * w, lm[6].y * h]) if lm[6].visibility > 0.5 else None
            r_elbow = np.array([lm[8].x * w, lm[8].y * h]) if lm[8].visibility > 0.5 else None
            r_wrist = np.array([lm[10].x * w, lm[10].y * h]) if lm[10].visibility > 0.5 else None
            
            if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
                elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            else:
                elbow_angle = 0
            elbow_angles.append(elbow_angle)
        else:
            elbow_angles.append(0)
    
    # Call segment_phases for phase segmentation
    # Returns: (labels, report) where report is a dict with boundaries
    print(f"\n[PHASE] Segmenting phases...")
    phase_labels, report = segment_phases(
        wrist_positions=wrist_positions,
        wrist_velocities=velocities["wrist_velocity_seq"],
        hip_velocities=velocities["hip_velocity_seq"],
        knee_angles=knee_angles,
        fps=fps,
        hip_positions=hip_positions,
        ankle_positions=ankle_positions,
        elbow_angles=elbow_angles,
    )
    
    # Extract boundaries from report
    boundaries = report.get("boundaries", {})
    
    # Save phase report
    report_path = os.path.join(output_dir, f"{session['session_id']}_phase_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[PHASE] Report saved to {report_path}")
    
    # Extract boundaries dict
    p_ranges = boundaries
    
    # Convert labels array to phase_map dict for compatibility
    phase_map = {i: phase_labels[i] for i in range(len(phase_labels))}

    print("\n-- Step 5b: Analyze run-up ----------------------------------")
    runup_output_path = os.path.join(output_dir, f"{session['session_id']}_runup_analysis.png")
    
    # Check if PNG already exists
    if os.path.exists(runup_output_path):
        print(f"\n[WARNING] Run-up PNG already exists: {runup_output_path}")
        response = input("Overwrite existing PNG? (yes/no): ").strip().lower()
        if response != "yes":
            print(f"[INFO] Skipping PNG generation. Keeping existing file.")
            # Load empty metrics to continue pipeline
            runup_metrics = {
                "peak_momentum_frame": None,
                "backfoot_contact_frame": None,
                "frontfoot_contact_frame": None,
                "approach_angle_deg": 0.0,
                "average_gate_width_px": 0.0,
                "stride_count": 0,
                "peak_velocity_px_frame": 0.0,
                "momentum_at_backfoot": 0.0,
                "momentum_at_release": 0.0,
            }
        else:
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
    else:
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
    print(f"[RUNUP] Approach angle: {runup_metrics.get('approach_angle_deg', 0):.1f}deg")
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

    # Compute angles with view priority metadata for feature display
    from src.biomechanics.angle_calculator import compute_angles_with_view_priority
    angle_seq = []
    angle_metadata_seq = []
    for lm in landmarks_smoothed:
        if lm is not None:
            angles, metadata = compute_angles_with_view_priority(lm, w, h, camera_view=camera_angle_type)
            angle_seq.append(angles)
            angle_metadata_seq.append(metadata)
        else:
            angle_seq.append({})
            angle_metadata_seq.append({})
    
    angle_summary     = summarize_angle_sequence(angle_seq)
    angles_at_release = angle_seq[release] if release is not None else {}
    angle_metadata_at_release = angle_metadata_seq[release] if release is not None else {}

    print("\n-- Step 6: Build delivery record ----------------------------")
    delivery_number = 1
    
    # Add view selection metadata
    view_selector = ViewSelector()
    available_views = {
        "back": all_frames.get("back") is not None,
        "front": all_frames.get("front") is not None,
        "side": all_frames.get("side") is not None,
    }
    view_metadata = view_selector.get_view_metadata(phase_map, available_views)
    
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
        angle_metadata    = angle_metadata_at_release,
        camera_view       = camera,
        view_sync_info    = view_metadata.get("phase_views", {}),
    )

    warnings = validate_record(record)
    for msg in warnings:
        print(f"[WARNING] {msg}")

    print("\n-- Step 7: Save ----------------------------------------------")
    save_delivery(record, session["session_id"], single_video_mode=False)

    print("\n-- Step 8: Render annotated video ----------------------------")
    out_path = os.path.join(
        output_dir,
        f"{session['session_id']}_{camera}_analysis.mp4"
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
        lm = landmarks_smoothed[i]
        render_kpts = coco_landmarks_raw[i] if i < len(coco_landmarks_raw) else None
        phase = phase_map.get(i, "")
        angles = angle_seq[i]
        angle_metadata = angle_metadata_seq[i]

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
                        # Strict validation: must be numeric, not NaN, not Inf, and reasonable range
                        if (not np.isnan(val_float) and not np.isinf(val_float) and 
                            -360 <= val_float <= 360):  # Angles should be in reasonable range
                            angle_annotations[key] = val_float
                    except (TypeError, ValueError, OverflowError):
                        # Skip if value can't be converted to float
                        pass

        # Prepare velocity data for HUD
        frame_velocities = None

        # Draw skeleton with HUD showing all metrics
        out = draw_skeleton(
            frame, render_kpts if render_kpts is not None else lm, w, h,
            angle_annotations=angle_annotations,
            phase_label=phase,
            bowling_style=frame_style,
            velocities=frame_velocities,
            angle_metadata=angle_metadata,
            frame_number=i,
        )
        out = _draw_wrist_trail(out, coco_landmarks_raw, i, wrist_idx=10, history=10)

        annotated.append(out)

    out_path = os.path.join(
        output_dir,
        f"{session['session_id']}_{camera}_analysis.mp4"
    )
    success = write_video(annotated, out_path, fps, slowdown=0.25)
    if success:
        print(f"\n[DONE] Output saved to: {out_path}")
    else:
        print(f"\n[ERROR] Failed to save video to: {out_path}")
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
    parser = argparse.ArgumentParser(description="Cricket Bowling Analysis with Manual Sync")
    parser.add_argument("--session", 
                        default=CONFIG["paths"]["input_four_camera"],
                        help=f"Path to session folder (default: {CONFIG['paths']['input_four_camera']})")
    parser.add_argument("--camera",  default="side",
                        choices=["front", "back", "side", "left", "right"])
    parser.add_argument("--no-ball", action="store_true",
                        help="Skip ball tracking")
    parser.add_argument("--output",  default=CONFIG["paths"].get("outputs_single", "outputs"))
    parser.add_argument("--sync-file", default=None,
                        help="Optional path to pre-saved sync offsets file")
    args = parser.parse_args()

    run_pipeline(
        session_path = args.session,
        camera       = args.camera,
        run_ball     = not args.no_ball,
        output_dir   = args.output,
        sync_file    = args.sync_file,
    )


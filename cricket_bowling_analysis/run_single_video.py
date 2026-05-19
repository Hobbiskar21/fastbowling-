"""
run_single_video.py
-------------------
Simple entry point for single video analysis.
Works with the current input_videos/single/ folder structure.

USAGE:
    python run_single_video.py
    (will ask which video to run)
    
    OR
    
    python run_single_video.py --video input_videos/single/bowling_video1.mp4
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
import csv
import yaml
import numpy as np

# Windows fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.pose import detect_pose_sequence as detect_pose_yolo, smooth_coco_keypoints_sequence, SmoothedLandmark
from src.biomechanics.angle_calculator import compute_all_angles, summarize_angle_sequence, compute_angles_with_view_priority
from src.biomechanics.velocity_estimator import compute_all_velocities
from src.biomechanics.phase_segmenter import segment_phases
from src.biomechanics.release_detector import find_release_frame
from src.biomechanics.feature_aggregator import build_delivery_record, validate_record
from src.biomechanics.bowling_style_detector import classify_bowling_style, get_camera_angle_type, validate_camera_angle
from src.biomechanics.runup_analyser import analyse_runup
from src.biomechanics.view_selector import ViewSelector
from src.storage.csv_writer import save_delivery
from src.visualization.skeleton_drawer import draw_skeleton
from src.utils.video_utils import write_video, get_video_info
from src.utils.config_loader import get_config
import cv2


def _get_workspace_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.dirname(project_root)


def _resolve_output_root(output_dir: str = None) -> str:
    workspace_root = _get_workspace_root()
    if not output_dir:
        return os.path.join(workspace_root, "outputs")
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(workspace_root, output_dir)


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


def _safe_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if np.isnan(value) or np.isinf(value):
        return ""
    return value


def _save_frame_data_csv(
    output_dir,
    session_id,
    raw_keypoints,
    smoothed_keypoints,
    phase_map,
    angle_seq,
    velocities,
    release_frame,
):
    """Save one row per frame so render/debug data matches the video timeline."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{session_id}_frame_data.csv")

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    angle_names = sorted({name for frame_angles in angle_seq for name in frame_angles.keys()})
    max_frames = max(len(raw_keypoints), len(smoothed_keypoints), len(angle_seq), len(phase_map))

    fieldnames = [
        "session_id", "frame", "phase", "is_release_frame",
        "wrist_velocity", "hip_velocity",
    ]
    for prefix in ("raw", "smoothed"):
        for name in keypoint_names:
            fieldnames.extend([f"{prefix}_{name}_x", f"{prefix}_{name}_y"])
    fieldnames.extend(angle_names)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        wrist_velocity_seq = velocities.get("wrist_velocity_seq", [])
        hip_velocity_seq = velocities.get("hip_velocity_seq", [])

        for frame_idx in range(max_frames):
            row = {
                "session_id": session_id,
                "frame": frame_idx,
                "phase": phase_map.get(frame_idx, ""),
                "is_release_frame": int(release_frame is not None and frame_idx == release_frame),
                "wrist_velocity": _safe_value(wrist_velocity_seq[frame_idx]) if frame_idx < len(wrist_velocity_seq) else "",
                "hip_velocity": _safe_value(hip_velocity_seq[frame_idx]) if frame_idx < len(hip_velocity_seq) else "",
            }

            for prefix, sequence in (("raw", raw_keypoints), ("smoothed", smoothed_keypoints)):
                kpts = sequence[frame_idx] if frame_idx < len(sequence) else None
                for kpt_idx, name in enumerate(keypoint_names):
                    if kpts is not None and kpt_idx < len(kpts):
                        row[f"{prefix}_{name}_x"] = _safe_value(kpts[kpt_idx][0])
                        row[f"{prefix}_{name}_y"] = _safe_value(kpts[kpt_idx][1])
                    else:
                        row[f"{prefix}_{name}_x"] = ""
                        row[f"{prefix}_{name}_y"] = ""

            angles = angle_seq[frame_idx] if frame_idx < len(angle_seq) else {}
            for name in angle_names:
                row[name] = _safe_value(angles.get(name))

            writer.writerow(row)

    print(f"[SINGLE VIDEO] Frame-by-frame CSV saved: {csv_path}")
    return csv_path


def list_available_videos(video_dir: str = None) -> list:
    """List all .mp4 files in the video directory."""
    if video_dir is None:
        # Get the script directory: fast-bowling-ananlysis/cricket_bowling_analysis/run_single_video.py
        # Need to go up 2 levels to reach the root (alisha/)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # cricket_bowling_analysis/
        project_root = os.path.dirname(script_dir)  # fast-bowling-ananlysis/
        workspace_root = os.path.dirname(project_root)  # alisha/ (root)
        video_dir = os.path.join(workspace_root, "input_videos", "single")
    
    if not os.path.exists(video_dir):
        print(f"[ERROR] Video directory not found: {video_dir}")
        return []
    
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    return sorted(videos)


def select_video_interactive(video_dir: str = None) -> str:
    """Ask user to select a video from available options."""
    if video_dir is None:
        # Get the script directory: fast-bowling-ananlysis/cricket_bowling_analysis/run_single_video.py
        # Need to go up 2 levels to reach the root (alisha/)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # cricket_bowling_analysis/
        project_root = os.path.dirname(script_dir)  # fast-bowling-ananlysis/
        workspace_root = os.path.dirname(project_root)  # alisha/ (root)
        video_dir = os.path.join(workspace_root, "input_videos", "single")
    
    videos = list_available_videos(video_dir)
    
    if not videos:
        print(f"[ERROR] No .mp4 files found in {video_dir}")
        sys.exit(1)
    
    print(f"\n[SINGLE VIDEO] Available videos in {video_dir}:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    
    while True:
        try:
            choice = input(f"\nSelect video (1-{len(videos)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                selected = os.path.join(video_dir, videos[idx])
                print(f"[SINGLE VIDEO] Selected: {videos[idx]}")
                return selected
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print(f"[ERROR] Please enter a valid number")


def run_single_video(video_path: str, output_dir: str = None) -> dict:
    """
    Analyze a single video file.
    
    Args:
        video_path: Path to .mp4 file
        output_dir: Where to save outputs (default: from config)
    
    Returns:
        DeliveryRecord dict
    """
    output_root = _resolve_output_root(output_dir)
    video_output_dir = os.path.join(output_root, "videos")
    artifact_output_dir = os.path.join(output_root, "artifacts")
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(artifact_output_dir, exist_ok=True)
    
    # Get video info
    print(f"\n[SINGLE VIDEO] Loading: {video_path}")
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    frame_count = info["frame_count"]
    
    print(f"[SINGLE VIDEO] Resolution: {w}x{h} | FPS: {fps} | Frames: {frame_count}")
    
    # Extract video name for session ID
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    session_id = f"single_{video_name}"
    
    # Extract frames
    print(f"\n[SINGLE VIDEO] Extracting frames...")
    from src.ingestion.frame_extractor import extract_single_camera
    original_frames, blurred_frames = extract_single_camera(video_path, offset=0)
    print(f"[SINGLE VIDEO] Extracted {len(original_frames)} frames")
    
    # Detect pose using the same original-frame flow as the reference scripts
    print(f"\n[SINGLE VIDEO] Detecting pose with YOLO...")
    coco_landmarks_raw, pose_metadata = detect_pose_yolo(
        original_frames, w, h,
        pose_model="yolov8m-pose.pt",
        imgsz=960,
        max_jump_px=260.0,
        device="cuda"  # Use GPU for faster inference
    )
    
    print(f"[SINGLE VIDEO] Smoothing COCO keypoints...")
    coco_landmarks_smoothed = smooth_coco_keypoints_sequence(coco_landmarks_raw)

    # Convert smoothed COCO keypoints to landmark objects for the rest of the pipeline
    print(f"[SINGLE VIDEO] Converting COCO keypoints to landmark objects...")
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
    print(f"[SINGLE VIDEO] Pose detection complete - YOLO with person tracking enabled")
    
    # Detect camera angle type for feature selection
    print(f"\n[SINGLE VIDEO] Detecting camera angle...")
    is_valid, camera_msg = validate_camera_angle(landmarks_smoothed, w, h)
    print(f"[INFO] {camera_msg}")
    
    camera_angle_type = get_camera_angle_type(landmarks_smoothed, w, h)
    print(f"[INFO] Detected camera angle: {camera_angle_type}")
    
    if camera_angle_type == 'side_on':
        print("[INFO] Side-on angle detected - using side view angles")
    else:
        print("[INFO] Front-on angle detected - using front view angles")
    
    # Compute velocities
    print(f"\n[SINGLE VIDEO] Computing velocities...")
    velocities = compute_all_velocities(landmarks_smoothed, fps, w, h)
    
    # Extract positions for phase segmentation
    wrist_positions = []
    for lm in landmarks_smoothed:
        if lm is not None and lm[16].visibility > 0.5:
            wrist_positions.append((lm[16].x * w, lm[16].y * h))
        else:
            wrist_positions.append(None)
    
    # Extract knee angles for phase segmentation
    knee_angles = []
    for lm in landmarks_smoothed:
        if lm is not None:
            from src.biomechanics.angle_calculator import calculate_angle
            # COCO indices: 12=rhip, 14=rknee, 16=rank
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
        if lm is not None and lm[12].visibility > 0.3:  # Right hip (COCO index 12)
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

    # Segment phases using updated phase_segmenter
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
    report_path = os.path.join(artifact_output_dir, f"{session_id}_phase_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[PHASE] Report saved to {report_path}")
    phase_map = {i: phase_labels[i] for i in range(len(phase_labels))}
    
    # Extract boundaries
    p_ranges = boundaries
    
    # Run-up analysis
    print(f"\n[SINGLE VIDEO] Analyzing run-up...")
    runup_output_path = os.path.join(artifact_output_dir, f"{session_id}_runup_analysis.png")
    
    if os.path.exists(runup_output_path):
        print(f"[WARNING] PNG already exists: {runup_output_path}")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
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
                landmarks_smoothed, phase_map, fps, w, h, runup_output_path,
                velocity_drop_threshold=0.3, gate_warn_multiplier=1.5, camera_angle_type=camera_angle_type
            )
    else:
        runup_metrics = analyse_runup(
            landmarks_smoothed, phase_map, fps, w, h, runup_output_path,
            velocity_drop_threshold=0.3, gate_warn_multiplier=1.5, camera_angle_type=camera_angle_type
        )
    
    # Find release frame
    release = find_release_frame(
        velocities["wrist_velocity_seq"],
        wrist_positions=wrist_positions,
        phase_map=phase_map,
    )
    print(f"[SINGLE VIDEO] Release frame: {release}")
    
    # Classify bowling style
    if release is not None and release < len(landmarks_smoothed):
        style, style_breakdown = classify_bowling_style(landmarks_smoothed[release], w, h)
        print(f"[SINGLE VIDEO] Bowling style: {style}")
    else:
        style = "UNKNOWN"
        style_breakdown = {'style': 'UNKNOWN', 'front_score': 0.0, 'side_score': 0.0, 'mixed_score': 0.0, 'signals': {}}
    
    # Compute angles
    print(f"\n[SINGLE VIDEO] Computing angles...")
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
    
    angle_summary = summarize_angle_sequence(angle_seq)
    angles_at_release = angle_seq[release] if release is not None else {}
    angle_metadata_at_release = angle_metadata_seq[release] if release is not None else {}
    
    # Build delivery record
    print(f"\n[SINGLE VIDEO] Building delivery record...")
    
    # Add view selection metadata
    view_selector = ViewSelector()
    available_views = {"back": False, "front": False, "side": True}  # Single video = side view only
    view_metadata = view_selector.get_view_metadata(phase_map, available_views)
    
    record = build_delivery_record(
        session_id=session_id,
        delivery_number=1,
        bowler_name=video_name,
        fps=fps,
        phase_map=phase_map,
        phase_ranges=p_ranges,
        release_frame=release,
        angles_at_release=angles_at_release,
        angle_summary=angle_summary,
        velocities=velocities,
        bowling_style=style,
        bowling_style_breakdown=style_breakdown,
        runup_metrics=runup_metrics,
        angle_metadata=angle_metadata_at_release,
        camera_view=camera_angle_type,
        view_sync_info=view_metadata.get("phase_views", {}),
    )
    
    warnings = validate_record(record)
    for msg in warnings:
        print(f"[WARNING] {msg}")
    
    # Save to CSV
    print(f"\n[SINGLE VIDEO] Saving to CSV...")
    save_delivery(record, session_id, single_video_mode=True)
    _save_frame_data_csv(
        output_dir=artifact_output_dir,
        session_id=session_id,
        raw_keypoints=coco_landmarks_raw,
        smoothed_keypoints=coco_landmarks_smoothed,
        phase_map=phase_map,
        angle_seq=angle_seq,
        velocities=velocities,
        release_frame=release,
    )
    
    # Render video
    print(f"\n[SINGLE VIDEO] Rendering annotated video...")
    out_path = os.path.join(video_output_dir, f"{session_id}_analysis.mp4")
    
    if os.path.exists(out_path):
        print(f"[WARNING] Video already exists: {out_path}")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print(f"[SINGLE VIDEO] Keeping existing video")
            return record
    
    annotated = []
    for i, frame in enumerate(original_frames):
        lm = landmarks_smoothed[i]
        render_kpts = coco_landmarks_raw[i] if i < len(coco_landmarks_raw) else None
        phase = phase_map.get(i, "")
        angles = angle_seq[i]
        angle_metadata = angle_metadata_seq[i]
        frame_style = style if i == release else None
        
        # Prepare angle annotations
        angle_annotations = {}
        if angles:
            for key, value in angles.items():
                if value is not None:
                    try:
                        val_float = float(value)
                        if (not np.isnan(val_float) and not np.isinf(val_float) and 
                            -360 <= val_float <= 360):
                            angle_annotations[key] = val_float
                    except (TypeError, ValueError, OverflowError):
                        pass
        
        # Draw skeleton on ORIGINAL frame (not blurred)
        out = draw_skeleton(
            frame, render_kpts if render_kpts is not None else lm, w, h,
            angle_annotations=angle_annotations,
            phase_label=phase,
            bowling_style=frame_style,
            velocities=None,
            angle_metadata=angle_metadata,
            frame_number=i,
        )
        out = _draw_wrist_trail(out, coco_landmarks_raw, i, wrist_idx=10, history=10)
        annotated.append(out)
    
    video_ok = write_video(annotated, out_path, fps, slowdown=0.25)
    if video_ok:
        print(f"[SINGLE VIDEO] Video saved: {out_path}")
    else:
        print(f"[ERROR] Video rendering failed: {out_path}")
    
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Video Analysis")
    parser.add_argument("--video", default=None, help="Path to .mp4 file (optional)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # If video not provided, ask user to select
    if args.video is None:
        video_path = select_video_interactive()
    else:
        video_path = args.video
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)
    
    run_single_video(video_path, args.output)

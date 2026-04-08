"""
src/biomechanics/runup_analyser.py
-----------------------------------
Analyzes bowler's run-up phase and generates comprehensive visual report.
Extracts hip path, velocity, momentum, foot contacts, gate pattern, and approach angle.
Outputs a single PNG with 5 panels showing complete run-up biomechanics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks


# MediaPipe landmark indices
L_HIP = 23
R_HIP = 24
L_ANKLE = 27
R_ANKLE = 28
L_SHOULDER = 11
R_SHOULDER = 12


def _extract_hip_path(landmarks_sequence, phase_map, width, height):
    """
    Extract hip midpoint (x, y) for every frame during RUN-UP phase.
    
    Returns:
        list of (x, y) tuples in pixels
        list of frame indices for RUN-UP phase
    """
    hip_path = []
    runup_frames = []
    
    for frame_idx, lm in enumerate(landmarks_sequence):
        # Only process RUN-UP phase
        if phase_map.get(frame_idx) != "RUN-UP":
            continue
        
        if lm is None:
            continue
        
        # Extract left and right hip
        if L_HIP < len(lm) and R_HIP < len(lm):
            l_hip = lm[L_HIP]
            r_hip = lm[R_HIP]
            
            if l_hip.visibility >= 0.5 and r_hip.visibility >= 0.5:
                # Calculate hip midpoint
                hip_x = (l_hip.x + r_hip.x) / 2 * width
                hip_y = (l_hip.y + r_hip.y) / 2 * height
                
                hip_path.append((hip_x, hip_y))
                runup_frames.append(frame_idx)
    
    return hip_path, runup_frames


def _calculate_velocities(hip_path, fps):
    """
    Calculate hip velocity per frame using central difference.
    v[t] = (pos[t+1] - pos[t-1]) / 2 / (1/fps)
    
    Returns:
        list of velocities in pixels per frame
    """
    velocities = []
    
    for i in range(len(hip_path)):
        if i == 0 or i == len(hip_path) - 1:
            # Edge frames: use forward/backward difference
            if i == 0:
                dx = hip_path[1][0] - hip_path[0][0]
                dy = hip_path[1][1] - hip_path[0][1]
            else:
                dx = hip_path[-1][0] - hip_path[-2][0]
                dy = hip_path[-1][1] - hip_path[-2][1]
        else:
            # Central difference
            dx = hip_path[i + 1][0] - hip_path[i - 1][0]
            dy = hip_path[i + 1][1] - hip_path[i - 1][1]
        
        # Distance in pixels per frame
        speed = np.sqrt(dx**2 + dy**2) / 2
        velocities.append(speed)
    
    return velocities


def _calculate_momentum(velocities):
    """
    Calculate cumulative momentum (proxy = cumulative velocity).
    Represents energy building up during run-up.
    
    Returns:
        list of cumulative momentum values
    """
    momentum = np.cumsum(velocities)
    return momentum.tolist()


def _find_foot_contacts(landmarks_sequence, phase_map, runup_frames, velocity_drop_threshold=0.3):
    """
    Find backfoot and frontfoot contact frames using ankle velocity drops.
    
    Backfoot contact = first sudden velocity drop in back ankle after run-up starts
    Frontfoot contact = first sudden velocity drop in front ankle after backfoot contact
    
    Returns:
        backfoot_frame, frontfoot_frame (frame indices, or None if not found)
    """
    # Extract ankle velocities during run-up
    ankle_velocities_l = []
    ankle_velocities_r = []
    
    for frame_idx in runup_frames:
        lm = landmarks_sequence[frame_idx]
        if lm is None:
            ankle_velocities_l.append(None)
            ankle_velocities_r.append(None)
            continue
        
        if L_ANKLE < len(lm) and R_ANKLE < len(lm):
            l_ankle = lm[L_ANKLE]
            r_ankle = lm[R_ANKLE]
            
            if l_ankle.visibility >= 0.5:
                ankle_velocities_l.append(l_ankle.y)  # Use Y for vertical motion
            else:
                ankle_velocities_l.append(None)
            
            if r_ankle.visibility >= 0.5:
                ankle_velocities_r.append(r_ankle.y)
            else:
                ankle_velocities_r.append(None)
        else:
            ankle_velocities_l.append(None)
            ankle_velocities_r.append(None)
    
    # Calculate velocity drops (inverted for peak finding)
    backfoot_frame = None
    frontfoot_frame = None
    
    # Find backfoot contact (right ankle for right-handed bowler)
    if ankle_velocities_r.count(None) < len(ankle_velocities_r):
        r_vel_clean = np.array([v if v is not None else np.nan for v in ankle_velocities_r])
        # Invert to find peaks (which are velocity drops)
        inverted = -r_vel_clean
        peaks, _ = find_peaks(inverted, height=velocity_drop_threshold)
        if len(peaks) > 0:
            backfoot_frame = runup_frames[peaks[0]]
    
    # Find frontfoot contact (left ankle)
    if backfoot_frame is not None and ankle_velocities_l.count(None) < len(ankle_velocities_l):
        # Only search after backfoot contact
        search_start = runup_frames.index(backfoot_frame) + 1
        if search_start < len(ankle_velocities_l):
            l_vel_clean = np.array([v if v is not None else np.nan for v in ankle_velocities_l[search_start:]])
            inverted = -l_vel_clean
            peaks, _ = find_peaks(inverted, height=velocity_drop_threshold)
            if len(peaks) > 0:
                frontfoot_frame = runup_frames[search_start + peaks[0]]
    
    return backfoot_frame, frontfoot_frame


def _calculate_gate_pattern(landmarks_sequence, phase_map, runup_frames, width):
    """
    Calculate ankle separation (gate width) per frame during run-up.
    Gate width = horizontal distance between left and right ankles.
    
    Returns:
        list of gate widths in pixels
    """
    gate_widths = []
    
    for frame_idx in runup_frames:
        lm = landmarks_sequence[frame_idx]
        if lm is None:
            gate_widths.append(None)
            continue
        
        if L_ANKLE < len(lm) and R_ANKLE < len(lm):
            l_ankle = lm[L_ANKLE]
            r_ankle = lm[R_ANKLE]
            
            if l_ankle.visibility >= 0.5 and r_ankle.visibility >= 0.5:
                # Horizontal distance
                gate_width = abs(r_ankle.x - l_ankle.x) * width
                gate_widths.append(gate_width)
            else:
                gate_widths.append(None)
        else:
            gate_widths.append(None)
    
    return gate_widths


def _calculate_shoulder_width(landmarks_sequence, width):
    """
    Calculate average shoulder width from first 10 frames.
    Used as reference for gate pattern analysis.
    """
    shoulder_widths = []
    
    for i, lm in enumerate(landmarks_sequence[:10]):
        if lm is None:
            continue
        
        if L_SHOULDER < len(lm) and R_SHOULDER < len(lm):
            l_shoulder = lm[L_SHOULDER]
            r_shoulder = lm[R_SHOULDER]
            
            if l_shoulder.visibility >= 0.5 and r_shoulder.visibility >= 0.5:
                shoulder_width = abs(r_shoulder.x - l_shoulder.x) * width
                shoulder_widths.append(shoulder_width)
    
    return np.mean(shoulder_widths) if shoulder_widths else 0


def _calculate_approach_angle(hip_path, backfoot_frame, runup_frames):
    """
    Calculate approach angle using last 15 hip positions before backfoot contact.
    Fit a line and calculate angle relative to vertical (pitch direction).
    
    Returns:
        angle in degrees
    """
    if backfoot_frame is None or len(hip_path) < 15:
        return 0.0
    
    # Find index of backfoot contact in hip_path
    try:
        backfoot_idx = runup_frames.index(backfoot_frame)
    except ValueError:
        return 0.0
    
    # Get last 15 positions before backfoot contact
    start_idx = max(0, backfoot_idx - 15)
    positions = hip_path[start_idx:backfoot_idx + 1]
    
    if len(positions) < 2:
        return 0.0
    
    # Extract x and y coordinates
    x_coords = np.array([p[0] for p in positions])
    y_coords = np.array([p[1] for p in positions])
    
    # Fit a line: y = mx + c
    coeffs = np.polyfit(x_coords, y_coords, 1)
    slope = coeffs[0]
    
    # Calculate angle relative to vertical
    # Vertical line has undefined slope, so we use atan2
    angle_rad = np.arctan(slope)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-90 range
    angle_deg = abs(angle_deg)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg


def _count_strides(gate_widths, gate_threshold_px=50):
    """
    Count strides by detecting peaks in gate width pattern.
    Each peak = one stride.
    """
    if not gate_widths or gate_widths.count(None) == len(gate_widths):
        return 0
    
    # Clean data
    clean_widths = np.array([w if w is not None else np.nan for w in gate_widths])
    
    # Find peaks (strides)
    peaks, _ = find_peaks(clean_widths, height=gate_threshold_px)
    
    return len(peaks)


def analyse_runup(
    landmarks_sequence,
    phase_map,
    fps,
    width,
    height,
    output_path,
    velocity_drop_threshold=0.3,
    gate_warn_multiplier=1.5,
    camera_angle_type=None,
):
    """
    Analyze bowler's run-up and generate comprehensive visual report.
    Only generates visualization if camera angle is valid (front/back view).
    
    Parameters
    ----------
    landmarks_sequence : list
        Per-frame MediaPipe landmarks
    phase_map : dict
        Frame index → phase name mapping
    fps : float
        Frames per second
    width, height : int
        Frame dimensions in pixels
    output_path : str
        Path to save PNG output
    velocity_drop_threshold : float
        Threshold for detecting foot contacts (velocity drop)
    gate_warn_multiplier : float
        Multiplier for gate width warning threshold
    camera_angle_type : str, optional
        Camera angle type ('front_back', 'side_on', etc.)
        If 'side_on', visualization is skipped
    
    Returns
    -------
    dict
        Metrics: peak_momentum_frame, backfoot_contact_frame, frontfoot_contact_frame,
                 approach_angle_deg, average_gate_width_px, stride_count,
                 peak_velocity_px_frame, momentum_at_backfoot
    """
    
    # Skip visualization if camera angle is side-on (not suitable for run-up analysis)
    skip_visualization = camera_angle_type == 'side_on'
    
    # Step 1: Extract hip path during run-up
    hip_path, runup_frames = _extract_hip_path(landmarks_sequence, phase_map, width, height)
    
    if len(hip_path) < 2:
        return {
            "peak_momentum_frame": None,
            "backfoot_contact_frame": None,
            "frontfoot_contact_frame": None,
            "approach_angle_deg": 0.0,
            "average_gate_width_px": 0.0,
            "stride_count": 0,
            "peak_velocity_px_frame": 0.0,
            "momentum_at_backfoot": 0.0,
        }
    
    # Step 2: Calculate velocities
    velocities = _calculate_velocities(hip_path, fps)
    
    # Step 3: Calculate momentum
    momentum = _calculate_momentum(velocities)
    
    # Step 4: Find foot contacts
    backfoot_frame, frontfoot_frame = _find_foot_contacts(
        landmarks_sequence, phase_map, runup_frames, velocity_drop_threshold
    )
    
    # Step 5: Calculate gate pattern
    gate_widths = _calculate_gate_pattern(landmarks_sequence, phase_map, runup_frames, width)
    shoulder_width = _calculate_shoulder_width(landmarks_sequence, width)
    
    # Step 6: Calculate approach angle
    approach_angle = _calculate_approach_angle(hip_path, backfoot_frame, runup_frames)
    
    # Step 7: Count strides
    stride_count = _count_strides(gate_widths)
    
    # Calculate metrics
    peak_momentum_frame = runup_frames[np.argmax(momentum)] if momentum else None
    peak_velocity_idx = np.argmax(velocities) if velocities else 0
    peak_velocity_frame = runup_frames[peak_velocity_idx] if peak_velocity_idx < len(runup_frames) else None
    peak_velocity = max(velocities) if velocities else 0.0
    
    momentum_at_backfoot = 0.0
    if backfoot_frame is not None:
        try:
            bf_idx = runup_frames.index(backfoot_frame)
            momentum_at_backfoot = momentum[bf_idx] if bf_idx < len(momentum) else 0.0
        except ValueError:
            pass

    # Convert None values to np.nan for nanmean calculation
    gate_widths_clean = [w if w is not None else np.nan for w in gate_widths]
    average_gate_width = np.nanmean(gate_widths_clean) if gate_widths_clean else 0.0
    
    # Step 8: Create visualization (only if camera angle is valid)
    if not skip_visualization:
        _create_runup_visualization(
            hip_path, runup_frames, velocities, momentum, gate_widths, shoulder_width,
            backfoot_frame, frontfoot_frame, peak_momentum_frame, approach_angle,
            output_path, gate_warn_multiplier, width, height
        )
    else:
        print(f"[RUNUP] Skipping visualization (side-on camera angle not suitable for run-up analysis)")
    
    return {
        "peak_momentum_frame": peak_momentum_frame,
        "backfoot_contact_frame": backfoot_frame,
        "frontfoot_contact_frame": frontfoot_frame,
        "approach_angle_deg": approach_angle,
        "average_gate_width_px": average_gate_width,
        "stride_count": stride_count,
        "peak_velocity_px_frame": peak_velocity,
        "momentum_at_backfoot": momentum_at_backfoot,
    }


def _create_runup_visualization(
    hip_path, runup_frames, velocities, momentum, gate_widths, shoulder_width,
    backfoot_frame, frontfoot_frame, peak_momentum_frame, approach_angle,
    output_path, gate_warn_multiplier, video_width=None, video_height=None
):
    """
    Create 5-panel visualization of run-up analysis.
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 24))

    # Panel 1: Top-down pitch map
    ax1 = plt.subplot(5, 1, 1)
    _draw_pitch_map(ax1, hip_path, backfoot_frame, frontfoot_frame, approach_angle, runup_frames, video_width, video_height)
    
    # Panel 2: Momentum build-up
    ax2 = plt.subplot(5, 1, 2)
    _draw_momentum_graph(ax2, runup_frames, momentum, backfoot_frame, frontfoot_frame, peak_momentum_frame)
    
    # Panel 3: Velocity per frame
    ax3 = plt.subplot(5, 1, 3)
    _draw_velocity_graph(ax3, runup_frames, velocities)
    
    # Panel 4: Gate pattern
    ax4 = plt.subplot(5, 1, 4)
    _draw_gate_pattern(ax4, runup_frames, gate_widths, shoulder_width, gate_warn_multiplier)
    
    # Panel 5: Metrics summary
    ax5 = plt.subplot(5, 1, 5)
    _draw_metrics_table(ax5, backfoot_frame, frontfoot_frame, approach_angle, shoulder_width, gate_widths)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()


def _draw_pitch_map(ax, hip_path, backfoot_frame, frontfoot_frame, approach_angle, runup_frames, video_width=None, video_height=None):
    """Panel 1: Realistic cricket field from wicket POV with run-up path (bowler at bottom)."""
    ax.set_title("Run-up Path (Wicket POV - Bowler at Bottom)", fontsize=14, color="white", fontweight="bold")
    
    # Cricket field dimensions (meters)
    pitch_width = 3.66  # 12 feet
    pitch_length = 20.12  # 66 feet
    outfield_extension = 10  # meters beyond pitch
    
    # Draw outfield (dark green background)
    outfield = patches.Rectangle(
        (-outfield_extension, -outfield_extension),
        pitch_width + 2 * outfield_extension,
        pitch_length + 2 * outfield_extension,
        linewidth=0, facecolor="#1a4d1a", zorder=1
    )
    ax.add_patch(outfield)
    
    # Draw pitch (cream/yellow)
    pitch = patches.Rectangle(
        (0, 0), pitch_width, pitch_length,
        linewidth=2, edgecolor="white", facecolor="#f4e4c1", zorder=2
    )
    ax.add_patch(pitch)
    
    # Draw crease lines (white)
    # Bowler's crease (at y=0, bottom)
    ax.plot([0, pitch_width], [0, 0], "w-", linewidth=1.5, alpha=0.7)
    # Popping crease (batter's end, at y=pitch_length, top)
    ax.plot([0, pitch_width], [pitch_length, pitch_length], "w-", linewidth=1.5, alpha=0.7)
    # Return creases
    ax.plot([0, 0], [0, pitch_length], "w-", linewidth=1, alpha=0.5)
    ax.plot([pitch_width, pitch_width], [0, pitch_length], "w-", linewidth=1, alpha=0.5)
    
    # Draw stumps at both ends with distinct colors
    stump_x = pitch_width / 2
    # Bowler's stumps (RED) - at bottom (y=0)
    ax.plot([stump_x], [0.5], "o", color="red", markersize=14, label="Bowler's Stumps", zorder=5)
    ax.text(stump_x - 0.5, -1.5, "BOWLER", color="red", fontsize=10, fontweight="bold", ha="center", zorder=5)
    
    # Batter's stumps (BLUE) - at top (y=pitch_length)
    ax.plot([stump_x], [pitch_length - 0.5], "o", color="cyan", markersize=14, label="Batter's Stumps", zorder=5)
    ax.text(stump_x - 0.5, pitch_length + 1.5, "BATTER", color="cyan", fontsize=10, fontweight="bold", ha="center", zorder=5)
    
    # Convert hip path from pixels to cricket field coordinates
    if hip_path and len(hip_path) > 0:
        # Use provided video dimensions or fall back to standard
        vid_w = video_width if video_width else 1920
        vid_h = video_height if video_height else 1080

        # Normalize path to pitch coordinates
        normalized_path = []
        for x_px, y_px in hip_path:
            # Map pixel coordinates to cricket field
            # Lateral: center of frame = center of pitch
            x_field = (x_px / vid_w) * pitch_width if x_px < vid_w else pitch_width / 2
            # Depth: y increases toward batter (from 0 to pitch_length)
            # Flip so that y_px=0 (top) maps to bowler's crease (y_field=0)
            y_field = (1 - y_px / vid_h) * pitch_length if y_px < vid_h else 0
            normalized_path.append((x_field, y_field))

        # Draw run-up path with gradient coloring (shows progression)
        if len(normalized_path) > 1:
            x_coords = [p[0] for p in normalized_path]
            y_coords = [p[1] for p in normalized_path]

            # Draw path segments with color gradient (blue start → yellow middle → red end)
            path_length = len(normalized_path)
            for i in range(len(normalized_path) - 1):
                progress = i / max(1, path_length - 1)  # 0 to 1

                # Create gradient: blue (0) → cyan (0.3) → yellow (0.6) → red (1)
                if progress < 0.3:
                    ratio = progress / 0.3
                    r, g, b = 0, ratio, 1
                elif progress < 0.6:
                    ratio = (progress - 0.3) / 0.3
                    r, g, b = ratio, 1, 1 - ratio
                else:
                    ratio = (progress - 0.6) / 0.4
                    r, g, b = 1, 1 - ratio, 0

                ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]],
                       color=(r, g, b), linewidth=5, zorder=4, alpha=0.85)

            # Add stride markers at regular intervals (every 5 points or fewer)
            stride_interval = max(1, len(normalized_path) // 6)
            for i in range(0, len(normalized_path), stride_interval):
                x, y = normalized_path[i]
                ax.plot([x], [y], "o", color="white", markersize=8, markeredgecolor="yellow",
                       markeredgewidth=1.5, zorder=5, alpha=0.8)

            # Highlight start point
            ax.plot([x_coords[0]], [y_coords[0]], "o", color="blue", markersize=14,
                   markeredgecolor="white", markeredgewidth=2, label="Run-up Start", zorder=6)

            # Highlight end point (delivery)
            ax.plot([x_coords[-1]], [y_coords[-1]], "o", color="red", markersize=14,
                   markeredgecolor="yellow", markeredgewidth=2, label="Delivery", zorder=6)
            ax.text(x_coords[-1] - 1, y_coords[-1] + 1, "DELIVERY", color="yellow",
                   fontsize=10, fontweight="bold", zorder=6)
    
    # Mark backfoot contact (red dot with border)
    if backfoot_frame is not None and hip_path:
        try:
            bf_idx = runup_frames.index(backfoot_frame)
            if bf_idx < len(hip_path):
                x_px, y_px = hip_path[bf_idx]
                vid_w = video_width if video_width else 1920
                vid_h = video_height if video_height else 1080
                x_field = (x_px / vid_w) * pitch_width
                y_field = (1 - y_px / vid_h) * pitch_length
                ax.plot([x_field], [y_field], "o", color="red", markersize=12, markeredgecolor="white",
                       markeredgewidth=2, label="Backfoot Contact", zorder=5)
                ax.text(x_field + 0.4, y_field - 0.6, "BF", color="red", fontsize=9, fontweight="bold", zorder=5)
        except (ValueError, IndexError):
            pass
    
    # Mark frontfoot contact (lime dot with border)
    if frontfoot_frame is not None and hip_path:
        try:
            ff_idx = runup_frames.index(frontfoot_frame)
            if ff_idx < len(hip_path):
                x_px, y_px = hip_path[ff_idx]
                vid_w = video_width if video_width else 1920
                vid_h = video_height if video_height else 1080
                x_field = (x_px / vid_w) * pitch_width
                y_field = (1 - y_px / vid_h) * pitch_length
                ax.plot([x_field], [y_field], "o", color="lime", markersize=12, markeredgecolor="white",
                       markeredgewidth=2, label="Frontfoot Contact", zorder=5)
                ax.text(x_field + 0.4, y_field + 0.6, "FF", color="lime", fontsize=9, fontweight="bold", zorder=5)
        except (ValueError, IndexError):
            pass
    
    # Draw approach angle visualization at bowler's crease
    if backfoot_frame is not None and hip_path:
        try:
            bf_idx = runup_frames.index(backfoot_frame)
            if bf_idx < len(hip_path):
                x_px, y_px = hip_path[bf_idx]
                vid_w = video_width if video_width else 1920
                vid_h = video_height if video_height else 1080
                x_delivery = (x_px / vid_w) * pitch_width
                y_delivery = (1 - y_px / vid_h) * pitch_length

                # Draw from bowler's crease (y=0, center of pitch)
                crease_x = pitch_width / 2
                crease_y = 0

                # Reference line (straight approach along pitch center)
                straight_line_length = 3.0
                ax.plot([crease_x, crease_x], [crease_y, crease_y + straight_line_length],
                       "white", linewidth=2, linestyle="--", alpha=0.5, label="Straight Approach", zorder=3)

                # Approach angle line (actual direction)
                angle_rad = np.radians(approach_angle)
                end_x = crease_x + straight_line_length * np.sin(angle_rad)
                end_y = crease_y + straight_line_length * np.cos(angle_rad)
                ax.plot([crease_x, end_x], [crease_y, end_y], color="gold", linewidth=3.5,
                       label=f"Approach Angle: {approach_angle:.1f}°", zorder=5)

                # Draw angle arc to show the deviation
                arc_radius = 1.2
                if approach_angle > 0:
                    angles = np.linspace(0, angle_rad, 30)
                    arc_x = crease_x + arc_radius * np.sin(angles)
                    arc_y = crease_y + arc_radius * np.cos(angles)
                    ax.plot(arc_x, arc_y, color="orange", linewidth=2, zorder=4)

                # Add angle text box
                text_x = crease_x + 1.5 * np.sin(angle_rad / 2)
                text_y = crease_y + 1.8
                ax.text(text_x, text_y, f"{approach_angle:.1f}°", fontsize=12, fontweight="bold",
                       color="gold", bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
                       ha="center", zorder=6)

                # Add delivery position marker
                ax.plot([x_delivery], [y_delivery], "D", color="gold", markersize=10,
                       markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        except (ValueError, IndexError):
            pass
    
    # Set limits with outfield visible
    ax.set_xlim(-outfield_extension, pitch_width + outfield_extension)
    ax.set_ylim(-outfield_extension, pitch_length + outfield_extension)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral Position (m)", color="white", fontsize=10)
    ax.set_ylabel("Pitch Direction (m) - Bowler at Bottom", color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95, title="Legend", title_fontsize=9)
    ax.grid(True, alpha=0.2, color="white")


def _draw_momentum_graph(ax, runup_frames, momentum, backfoot_frame, frontfoot_frame, peak_momentum_frame):
    """Panel 2: Momentum build-up graph with event markers."""
    ax.set_title("Momentum Build-up During Run-up", fontsize=14, color="white", fontweight="bold")
    
    ax.plot(runup_frames, momentum, "b-", linewidth=2, label="Cumulative Momentum")
    
    # Mark events
    if backfoot_frame is not None:
        try:
            bf_idx = runup_frames.index(backfoot_frame)
            ax.axvline(backfoot_frame, color="red", linestyle="--", linewidth=2, alpha=0.7)
            ax.text(backfoot_frame, max(momentum) * 0.9, "Backfoot", color="red", fontsize=9, rotation=90)
        except ValueError:
            pass
    
    if frontfoot_frame is not None:
        try:
            ff_idx = runup_frames.index(frontfoot_frame)
            ax.axvline(frontfoot_frame, color="green", linestyle="--", linewidth=2, alpha=0.7)
            ax.text(frontfoot_frame, max(momentum) * 0.85, "Frontfoot", color="green", fontsize=9, rotation=90)
        except ValueError:
            pass
    
    if peak_momentum_frame is not None:
        try:
            pm_idx = runup_frames.index(peak_momentum_frame)
            ax.axvline(peak_momentum_frame, color="orange", linestyle="--", linewidth=2, alpha=0.7)
            ax.text(peak_momentum_frame, max(momentum) * 0.95, "Peak", color="orange", fontsize=9, rotation=90)
        except ValueError:
            pass
    
    ax.set_xlabel("Frame Number", color="white")
    ax.set_ylabel("Cumulative Momentum", color="white")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)


def _draw_velocity_graph(ax, runup_frames, velocities):
    """Panel 3: Velocity per frame showing acceleration and deceleration."""
    ax.set_title("Hip Velocity During Run-up", fontsize=14, color="white", fontweight="bold")
    
    ax.plot(runup_frames, velocities, "c-", linewidth=2, label="Hip Speed")
    ax.fill_between(runup_frames, velocities, alpha=0.3, color="cyan")
    
    ax.set_xlabel("Frame Number", color="white")
    ax.set_ylabel("Speed (pixels/frame)", color="white")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)


def _draw_gate_pattern(ax, runup_frames, gate_widths, shoulder_width, gate_warn_multiplier):
    """Panel 4: Gate pattern (ankle separation) with shoulder width reference."""
    ax.set_title("Gate Pattern (Ankle Separation)", fontsize=14, color="white", fontweight="bold")
    
    # Clean data
    clean_widths = [w if w is not None else np.nan for w in gate_widths]
    ax.plot(runup_frames, clean_widths, "m-", linewidth=2, label="Gate Width")
    
    # Shoulder width reference
    ax.axhline(shoulder_width, color="yellow", linestyle="--", linewidth=2, alpha=0.7, label=f"Shoulder Width ({shoulder_width:.0f}px)")
    
    # Warning thresholds
    warn_threshold = shoulder_width * gate_warn_multiplier
    ax.axhline(warn_threshold, color="red", linestyle=":", linewidth=1, alpha=0.5, label=f"Warning Threshold ({warn_threshold:.0f}px)")
    
    # Shade warning regions
    ax.fill_between(runup_frames, warn_threshold, max(clean_widths) * 1.2, alpha=0.1, color="red")
    
    ax.set_xlabel("Frame Number", color="white")
    ax.set_ylabel("Ankle Separation (pixels)", color="white")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)


def _draw_metrics_table(ax, backfoot_frame, frontfoot_frame, approach_angle, shoulder_width, gate_widths):
    """Panel 5: Key metrics summary table."""
    ax.set_title("Run-up Metrics Summary", fontsize=14, color="white", fontweight="bold")
    ax.axis("off")

    # Calculate metrics - convert None to np.nan for nanmean
    gate_widths_clean = [w if w is not None else np.nan for w in gate_widths]
    average_gate = np.nanmean(gate_widths_clean) if gate_widths_clean else 0.0
    stride_count = _count_strides(gate_widths)
    
    # Create table data
    metrics = [
        ["Metric", "Value"],
        ["Approach Angle", f"{approach_angle:.1f}°"],
        ["Backfoot Contact Frame", f"{backfoot_frame}" if backfoot_frame else "N/A"],
        ["Frontfoot Contact Frame", f"{frontfoot_frame}" if frontfoot_frame else "N/A"],
        ["Average Gate Width", f"{average_gate:.1f} px"],
        ["Shoulder Width", f"{shoulder_width:.1f} px"],
        ["Gate/Shoulder Ratio", f"{average_gate/shoulder_width:.2f}" if shoulder_width > 0 else "N/A"],
        ["Stride Count", f"{stride_count}"],
    ]
    
    # Draw table
    table = ax.table(cellText=metrics, cellLoc="left", loc="center", 
                     colWidths=[0.5, 0.5], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor("#1f77b4")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Alternate row colors
    for i in range(1, len(metrics)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#2a2a2a")
            else:
                table[(i, j)].set_facecolor("#1a1a1a")
            table[(i, j)].set_text_props(color="white")

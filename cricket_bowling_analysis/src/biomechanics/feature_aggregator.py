"""
src/biomechanics/feature_aggregator.py
----------------------------------------
Combines all biomechanics outputs into a single flat DeliveryRecord dict.
This is the one object passed to storage (CSV, Parquet, or PostgreSQL).

To add a new feature:
    1. Compute it anywhere in the pipeline
    2. Pass it into build_delivery_record()
    3. Add the key to the returned dict
"""

from typing import Optional


def build_delivery_record(session_id: str,
                           delivery_number: int,
                           bowler_name: str,
                           fps: float,
                           phase_map: dict,
                           phase_ranges: dict,
                           release_frame: Optional[int],
                           angles_at_release: dict,
                           angle_summary: dict,
                           velocities: dict,
                           bowling_style: Optional[str] = None,
                           bowling_style_breakdown: Optional[dict] = None,
                           runup_metrics: Optional[dict] = None) -> dict:
    """
    Build the complete DeliveryRecord for one delivery.

    Returns:
        Flat dict ready for CSV, Parquet, or PostgreSQL.
    """
    if bowling_style_breakdown is None:
        bowling_style_breakdown = {}
    if runup_metrics is None:
        runup_metrics = {}
    
    return {
        # Identity
        "session_id":       session_id,
        "delivery_number":  delivery_number,
        "bowler_name":      bowler_name,

        # Release
        "release_frame":        release_frame,

        # Angles at release frame
        "elbow_angle_at_release":       angles_at_release.get("elbow_angle"),
        "shoulder_angle_at_release":    angles_at_release.get("shoulder_angle"),
        "front_knee_angle_at_release":  angles_at_release.get("front_knee_angle"),
        "back_knee_angle_at_release":   angles_at_release.get("back_knee_angle"),
        "hip_angle_at_release":         angles_at_release.get("hip_angle"),
        "trunk_lean_at_release":        angles_at_release.get("trunk_lean"),
        "hip_shoulder_sep_at_release":  angles_at_release.get("hip_shoulder_sep"),

        # Peak angles across full delivery
        "elbow_angle_max":       angle_summary.get("elbow_angle", {}).get("max"),
        "shoulder_angle_max":    angle_summary.get("shoulder_angle", {}).get("max"),
        "front_knee_angle_min":  angle_summary.get("front_knee_angle", {}).get("min"),
        "hip_shoulder_sep_max":  angle_summary.get("hip_shoulder_sep", {}).get("max"),
        "trunk_lean_max":        angle_summary.get("trunk_lean", {}).get("max"),

        # Velocities
        "arm_velocity_max":   velocities.get("arm_velocity_max"),
        "arm_velocity_mean":  velocities.get("arm_velocity_mean"),
        "runup_speed_mean":   velocities.get("runup_speed_mean"),

        # Bowling Style
        "bowling_style":        bowling_style,
        "bowling_style_front_score": bowling_style_breakdown.get("front_score"),
        "bowling_style_side_score":  bowling_style_breakdown.get("side_score"),

        # Run-up Metrics
        "runup_peak_momentum_frame":      runup_metrics.get("peak_momentum_frame"),
        "runup_backfoot_contact_frame":   runup_metrics.get("backfoot_contact_frame"),
        "runup_frontfoot_contact_frame":  runup_metrics.get("frontfoot_contact_frame"),
        "runup_approach_angle_deg":       runup_metrics.get("approach_angle_deg"),
        "runup_average_gate_width_px":    runup_metrics.get("average_gate_width_px"),
        "runup_stride_count":             runup_metrics.get("stride_count"),
        "runup_peak_velocity_px_frame":   runup_metrics.get("peak_velocity_px_frame"),
        "runup_momentum_at_backfoot":     runup_metrics.get("momentum_at_backfoot"),

        # Phase frame ranges
        "phase_runup_start":         phase_ranges.get("RUN-UP", {}).get("start"),
        "phase_runup_end":           phase_ranges.get("RUN-UP", {}).get("end"),
        "phase_loadup_start":        phase_ranges.get("LOAD-UP", {}).get("start"),
        "phase_loadup_end":          phase_ranges.get("LOAD-UP", {}).get("end"),
        "phase_delivery_start":      phase_ranges.get("DELIVERY", {}).get("start"),
        "phase_delivery_end":        phase_ranges.get("DELIVERY", {}).get("end"),
        "phase_followthrough_start": phase_ranges.get("FOLLOW-THROUGH", {}).get("start"),
        "phase_followthrough_end":   phase_ranges.get("FOLLOW-THROUGH", {}).get("end"),

        # ADD NEW FEATURES HERE
    }


def validate_record(record: dict) -> list:
    """
    Check for missing critical fields.

    Returns:
        List of warning strings. Empty = record is clean.
    """
    warnings = []
    critical = ["release_frame", "elbow_angle_at_release",
                "arm_velocity_max", "runup_speed_mean"]
    for field in critical:
        if record.get(field) is None:
            warnings.append(f"Missing critical field: {field}")
    return warnings
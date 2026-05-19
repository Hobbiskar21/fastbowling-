"""
Basic repeatability scoring for bowling action analysis.

Calculates scores based on deviation from mean curves across multiple deliveries.
Simple scoring: low deviation = high repeatability.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from pathlib import Path

from .config import PHASE_WEIGHTS, PHASE_FEATURES


def calculate_repeatability_scores(
    curves_csv_paths: List[str],
    output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate repeatability scores from movement curves.

    For each phase and feature:
    - Combines curves from multiple deliveries
    - Calculates mean curve
    - Computes average deviation from mean
    - Converts to 0-100 score

    Args:
        curves_csv_paths: List of paths to movement curve CSVs.
        output_csv_path: Optional path to save scores CSV.

    Returns:
        DataFrame with feature-level, phase-level, and overall scores.
    """
    # Load all curves
    all_curves = []
    for csv_path in curves_csv_paths:
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            all_curves.append(df)

    if not all_curves:
        raise ValueError("No valid curve CSVs provided")

    combined_df = pd.concat(all_curves, ignore_index=True)
    print(f"Loaded {len(all_curves)} deliveries")

    # Scoring parameters by feature type
    # These are empirical scales; adjust based on actual data ranges
    feature_scales = {
        # Angle features (degrees)
        "front_knee_angle": 5.0,
        "back_knee_angle": 5.0,
        "trunk_lean_angle": 5.0,
        "bowling_arm_angle": 5.0,
        "bowling_elbow_angle": 5.0,
        "front_leg_angle": 5.0,
        "back_leg_angle": 5.0,
        # Position features (normalized, 0-1)
        "hip_center_x": 0.1,
        "hip_center_y": 0.1,
        "head_x": 0.1,
        "head_y": 0.1,
        "shoulder_center_x": 0.1,
        "shoulder_center_y": 0.1,
        "front_ankle_x": 0.1,
        "front_ankle_y": 0.1,
        "back_ankle_x": 0.1,
        "back_ankle_y": 0.1,
        "bowling_wrist_x": 0.1,
        "bowling_wrist_y": 0.1,
        "body_center_x": 0.1,
        "body_center_y": 0.1,
        # Velocity features
        "wrist_speed": 10.0,
        "hip_speed": 10.0,
        "head_speed": 10.0,
        "front_ankle_speed": 10.0,
        "front_knee_angle_velocity": 5.0,
        "arm_angle_velocity": 5.0,
        "trunk_lean_velocity": 5.0,
        # Proxy features
        "stride_length_proxy": 0.1,
        "release_height_proxy": 0.1,
    }

    score_rows = []
    phase_scores = {}

    # Process each phase
    phases = combined_df["phase"].unique()
    phases = [p for p in phases if p != "unassigned"]

    for phase_name in sorted(phases):
        phase_data = combined_df[combined_df["phase"] == phase_name]

        if phase_data.empty:
            continue

        # Get relevant features
        requested_features = PHASE_FEATURES.get(phase_name, [])
        numeric_cols = [f for f in requested_features if f in phase_data.columns]

        if not numeric_cols:
            continue

        phase_feature_scores = []

        # Score each feature
        for feature_name in numeric_cols:
            feature_data = phase_data[["normalized_index", feature_name]].copy()
            feature_data = feature_data.dropna(subset=[feature_name])

            if feature_data.empty:
                continue

            # Group by normalized_index and get mean
            grouped = feature_data.groupby("normalized_index")[feature_name].agg(['mean', 'std', 'count'])

            # Calculate mean absolute deviation
            deviations = []
            for idx, row in feature_data.iterrows():
                norm_idx = int(row["normalized_index"])
                value = row[feature_name]
                mean_value = grouped.loc[norm_idx, 'mean']
                deviation = abs(value - mean_value)
                deviations.append(deviation)

            if not deviations:
                continue

            avg_deviation = np.mean(deviations)

            # Convert to score using feature-specific scale
            scale = feature_scales.get(feature_name, 1.0)
            score = max(0, min(100, 100 - (avg_deviation * scale)))

            score_rows.append({
                "level": "feature",
                "phase": phase_name,
                "feature": feature_name,
                "deviation": round(avg_deviation, 4),
                "score": round(score, 2),
                "scale": scale
            })

            phase_feature_scores.append(score)

        # Calculate phase score
        if phase_feature_scores:
            phase_score = np.mean(phase_feature_scores)
            score_rows.append({
                "level": "phase",
                "phase": phase_name,
                "feature": None,
                "deviation": None,
                "score": round(phase_score, 2),
                "scale": None
            })
            phase_scores[phase_name] = phase_score

    # Calculate overall score
    if phase_scores:
        overall_score = 0.0
        weight_sum = 0.0
        for phase_name, score in phase_scores.items():
            weight = PHASE_WEIGHTS.get(phase_name, 0.0)
            overall_score += score * weight
            weight_sum += weight

        if weight_sum > 0:
            overall_score = overall_score / weight_sum

        score_rows.append({
            "level": "overall",
            "phase": None,
            "feature": None,
            "deviation": None,
            "score": round(overall_score, 2),
            "scale": None
        })

    scores_df = pd.DataFrame(score_rows)

    # Save if requested
    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(output_csv_path, index=False)
        print(f"Saved scores to: {output_csv_path}")

    # Print summary
    print("\n=== Repeatability Scores ===")
    for _, row in scores_df.iterrows():
        if row["level"] == "overall":
            print(f"\nOverall Score: {row['score']}/100")
        elif row["level"] == "phase":
            print(f"  {row['phase']}: {row['score']}/100")

    return scores_df

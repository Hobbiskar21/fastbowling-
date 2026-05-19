"""
Repeatability visualization for bowling action analysis.

Creates graphs comparing movements across multiple deliveries to visualize
consistency and identify variations in technique.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .config import PHASE_FEATURES, PHASE_NAMES


def plot_feature_curve_across_deliveries(
    curves_csv_paths: List[str],
    feature_name: str,
    phase_name: str,
    output_path: str,
    title_suffix: str = ""
) -> bool:
    """
    Plot a single feature across multiple deliveries for one phase.

    Args:
        curves_csv_paths: List of paths to movement curve CSVs.
        feature_name: Feature to plot (e.g., 'front_knee_angle').
        phase_name: Phase to plot (e.g., 'ffc_to_release').
        output_path: Path to save PNG graph.
        title_suffix: Optional suffix for graph title.

    Returns:
        True if successful, False if feature/phase not found.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    found_any = False
    deliveries = []

    for csv_path in curves_csv_paths:
        if not Path(csv_path).exists():
            print(f"Warning: CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Filter for phase and feature
        phase_data = df[df["phase"] == phase_name]

        if phase_data.empty:
            continue

        if feature_name not in phase_data.columns:
            continue

        found_any = True

        # Get delivery label
        if "delivery_id" in phase_data.columns:
            delivery_id = phase_data["delivery_id"].iloc[0]
        else:
            delivery_id = Path(csv_path).stem

        deliveries.append(delivery_id)

        # Get normalized time and feature values
        norm_time = phase_data["normalized_time"].values
        feature_values = phase_data[feature_name].values

        # Filter out NaN values
        mask = ~np.isnan(feature_values)
        if mask.any():
            ax.plot(norm_time[mask], feature_values[mask], marker='o', label=delivery_id, linewidth=2, markersize=4)

    if not found_any:
        print(f"No data found for feature '{feature_name}' in phase '{phase_name}'")
        return False

    # Labels and formatting
    ax.set_xlabel("Normalized Time", fontsize=12)
    ax.set_ylabel(feature_name, fontsize=12)
    title = f"{feature_name} during {phase_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

    print(f"Saved graph: {output_path}")
    return True


def plot_all_key_features(
    curves_csv_paths: List[str],
    output_dir: str
) -> None:
    """
    Plot all key features across all phases.

    Args:
        curves_csv_paths: List of paths to movement curve CSVs.
        output_dir: Directory to save PNG graphs.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define key feature plots by phase
    key_plots = {
        "approach": ["hip_speed", "trunk_lean_angle", "head_x"],
        "jump_bound": ["front_knee_angle", "back_knee_angle", "hip_center_y"],
        "bfc_window": ["back_ankle_y", "back_knee_angle", "trunk_lean_angle"],
        "bfc_to_ffc": ["hip_speed", "front_ankle_x", "back_ankle_x"],
        "ffc_window": ["front_knee_angle", "front_ankle_y", "hip_center_y"],
        "ffc_to_release": [
            "front_knee_angle",
            "bowling_arm_angle",
            "trunk_lean_angle",
            "wrist_speed",
            "bowling_elbow_angle",
            "head_y"
        ],
        "follow_through": ["hip_speed", "trunk_lean_angle", "head_x"],
    }

    for phase_name, features in key_plots.items():
        for feature_name in features:
            output_path = output_dir_path / f"{phase_name}_{feature_name}.png"
            plot_feature_curve_across_deliveries(
                curves_csv_paths,
                feature_name,
                phase_name,
                str(output_path),
                title_suffix="Repeatability Check"
            )

    print(f"Completed plotting to: {output_dir}")


def plot_phase_overlay(
    curves_csv_paths: List[str],
    phase_name: str,
    output_dir: str
) -> None:
    """
    Plot all features of a phase in a grid for detailed comparison.

    Args:
        curves_csv_paths: List of paths to movement curve CSVs.
        phase_name: Phase to plot.
        output_dir: Directory to save PNG graph.
    """
    # Get features for this phase
    features = PHASE_FEATURES.get(phase_name, [])
    if not features:
        print(f"No features defined for phase: {phase_name}")
        return

    # Read first CSV to check available features
    df = pd.read_csv(curves_csv_paths[0])
    available_features = [f for f in features if f in df.columns]

    if not available_features:
        print(f"No available features for phase: {phase_name}")
        return

    # Create grid
    n_cols = 3
    n_rows = (len(available_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if len(available_features) > 1 else [axes]

    for idx, feature_name in enumerate(available_features):
        ax = axes[idx]

        for csv_path in curves_csv_paths:
            if not Path(csv_path).exists():
                continue

            df = pd.read_csv(csv_path)
            phase_data = df[df["phase"] == phase_name]

            if phase_data.empty or feature_name not in phase_data.columns:
                continue

            if "delivery_id" in phase_data.columns:
                delivery_id = phase_data["delivery_id"].iloc[0]
            else:
                delivery_id = Path(csv_path).stem

            norm_time = phase_data["normalized_time"].values
            feature_values = phase_data[feature_name].values

            mask = ~np.isnan(feature_values)
            if mask.any():
                ax.plot(norm_time[mask], feature_values[mask], marker='o', label=delivery_id, linewidth=1.5, markersize=3)

        ax.set_title(feature_name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Normalized Time", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(available_features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Phase: {phase_name} - All Features", fontsize=14, fontweight='bold')
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{phase_name}_overlay.png"
    plt.savefig(str(output_path), dpi=100)
    plt.close()

    print(f"Saved phase overlay: {output_path}")


def plot_lstm_sequence_features(sequence_csv_path: str, output_path: str) -> bool:
    """Plot key LSTM features across the 120 timestep delivery sequence."""
    path = Path(sequence_csv_path)
    if not path.exists():
        print(f"Warning: sequence CSV not found: {path}")
        return False
    df = pd.read_csv(path)
    features = [
        "front_knee_angle",
        "trunk_lean_angle",
        "bowling_arm_angle",
        "wrist_speed",
        "hip_speed",
        "head_y",
    ]
    available = [feature for feature in features if feature in df.columns]
    if not available:
        print("Warning: no key LSTM features available for plotting.")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    for feature in available:
        ax.plot(x, pd.to_numeric(df[feature], errors="coerce").ffill().bfill().fillna(0), label=feature)

    boundary = 0
    fixed_lengths = [20, 15, 10, 20, 10, 25, 20]
    for length in fixed_lengths[:-1]:
        boundary += length
        ax.axvline(boundary, color="black", alpha=0.25, linestyle="--", linewidth=1)

    ax.set_title("LSTM Delivery Sequence Features")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feature value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved LSTM sequence plot: {output_path}")
    return True

"""Create a PNG dashboard for repeatability predictions."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .config import PHASE_DISPLAY_NAMES, PHASE_NAMES


def verdict_from_score(score: float) -> str:
    if score >= 80:
        return "High Repeatability Potential"
    if score >= 60:
        return "Moderate Repeatability Potential"
    return "Low Repeatability Potential"


def create_repeatability_dashboard(
    bowler_id,
    delivery_id,
    overall_score,
    phase_scores: Dict[str, float],
    output_path,
    graphs_dir=None,
):
    """Save a dashboard PNG with final and phase-wise scores."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    phase_values = [float(phase_scores.get(p, 0.0)) for p in PHASE_NAMES]
    display_names = [PHASE_DISPLAY_NAMES[p] for p in PHASE_NAMES]
    strongest = PHASE_NAMES[int(np.argmax(phase_values))]
    weakest = PHASE_NAMES[int(np.argmin(phase_values))]
    verdict = verdict_from_score(float(overall_score))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#d6dde5",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#334155",
    })
    fig = plt.figure(figsize=(12, 8), facecolor="#f8fafc")
    grid = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.05, 2.25], hspace=0.28)

    ax_title = fig.add_subplot(grid[0])
    ax_title.axis("off")
    ax_title.text(0.02, 0.72, "Side-on Repeatability Analysis", fontsize=20, fontweight="bold", color="#0f172a")
    ax_title.text(0.02, 0.24, f"Bowler: {bowler_id}   |   Delivery: {delivery_id}", fontsize=11, color="#64748b")

    ax_summary = fig.add_subplot(grid[1])
    ax_summary.axis("off")
    ax_summary.text(0.02, 0.66, f"{overall_score:.0f}", fontsize=46, fontweight="bold", color="#0f766e")
    ax_summary.text(0.145, 0.74, "/ 100", fontsize=16, color="#64748b")
    ax_summary.text(0.02, 0.24, "Repeatability score", fontsize=12, color="#475569")
    ax_summary.text(0.36, 0.66, verdict, fontsize=17, fontweight="bold", color="#1e293b")
    ax_summary.text(0.36, 0.36, f"Strongest: {PHASE_DISPLAY_NAMES[strongest]}", fontsize=12, color="#334155")
    ax_summary.text(0.36, 0.14, f"Needs attention: {PHASE_DISPLAY_NAMES[weakest]}", fontsize=12, color="#334155")
    ax_summary.text(
        0.68,
        0.46,
        "Use this as a consistency readout across the seven delivery phases.",
        fontsize=11,
        color="#64748b",
        wrap=True,
    )

    ax = fig.add_subplot(grid[2])
    y = np.arange(len(display_names))
    colors = ["#0f766e" if p == strongest else "#2563eb" if p != weakest else "#94a3b8" for p in PHASE_NAMES]
    ax.barh(y, phase_values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.invert_yaxis()
    for idx, value in enumerate(phase_values):
        ax.text(min(value + 1.5, 96), idx, f"{value:.0f}", va="center", fontsize=10, color="#0f172a")
    ax.grid(axis="x", alpha=0.16, color="#94a3b8")
    ax.set_axisbelow(True)
    ax.set_title("Phase Scores", loc="left", fontsize=13, fontweight="bold", color="#0f172a", pad=12)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.16, right=0.96, top=0.94, bottom=0.08)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"[REPEATABILITY] Saved dashboard: {output}")
    return output

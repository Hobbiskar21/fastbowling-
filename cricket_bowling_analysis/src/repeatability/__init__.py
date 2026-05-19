"""
Repeatability analysis module for side-on bowling action.

Provides phase detection, feature selection, movement analysis,
visualization, and basic scoring for bowling delivery repeatability.
"""
"""Side-on repeatability analysis package.

This package consumes frame-wise feature CSVs and builds repeatability
inputs, confirmed phases, movement curves, fixed 120-step LSTM sequences,
baseline scores, model predictions, and dashboard outputs.
"""

from .config import LSTM_FEATURES, PHASE_NAMES

__all__ = ["PHASE_NAMES", "LSTM_FEATURES"]

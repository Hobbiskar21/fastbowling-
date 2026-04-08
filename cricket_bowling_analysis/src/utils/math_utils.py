"""
src/utils/math_utils.py
-------------------------
Low-level math helpers shared across biomechanics modules.
"""

import numpy as np
from scipy.signal import savgol_filter


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at point B in triangle A-B-C."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def smooth_sequence(values: list, window: int = 5, polyorder: int = 2) -> list:
    """
    Apply Savitzky-Golay smoothing to a list of floats.
    Handles None values by interpolating before smoothing
    and restoring None at the same positions after.
    """
    indices_none = [i for i, v in enumerate(values) if v is None]
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)

    nans = np.isnan(arr)
    if nans.all():
        return values

    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    actual_window = min(window, len(arr))
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window < 3:
        return values

    smoothed = savgol_filter(arr, actual_window, polyorder)
    result = list(smoothed)
    for i in indices_none:
        result[i] = None
    return result


def pixels_to_metres(pixels: float, pixels_per_metre: float) -> float:
    return pixels / pixels_per_metre if pixels_per_metre > 0 else pixels


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
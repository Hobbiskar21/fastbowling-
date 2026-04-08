"""
src/biomechanics/phase_segmenter.py
--------------------------------------
Labels every frame as one of 4 bowling phases:
    RUN-UP         : static stance → first movement
    LOAD-GATHER    : movement start → gather/coil (jump)
    DELIVERY       : gather → peak wrist velocity (release)
    FOLLOW-THROUGH : after release
"""

import warnings
import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(data: ArrayLike, fill_value: float = 0.0) -> np.ndarray:
    """Convert list/array to float64 ndarray, replacing None with fill_value."""
    return np.array(
        [fill_value if v is None else v for v in data],
        dtype=np.float64,
    )


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Causal moving-average smoothing (no lookahead).
    Pads with the first value so the output is the same length as input.
    """
    if window <= 1 or len(signal) == 0:
        return signal.copy()
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, signal[0]), signal])
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Phase boundary detectors
# ---------------------------------------------------------------------------

def detect_movement_start(
    wrist_positions: ArrayLike,
    threshold: float = 5.0,
    consecutive: int = 3,
) -> int:
    """
    Returns the first frame where wrist displacement stays above *threshold*
    for *consecutive* frames in a row.  Requiring a run of frames prevents
    a single noisy reading from triggering the transition.

    Parameters
    ----------
    wrist_positions : array-like of (2,) or (3,) ndarrays
        Per-frame wrist position in pixels or metres.
    threshold : float
        Minimum inter-frame displacement to count as motion.
    consecutive : int
        How many consecutive above-threshold frames are required.

    Returns
    -------
    int
        Frame index of the movement start, or 1 if never found.
    """
    positions = [
        np.zeros(2) if p is None else np.asarray(p, dtype=np.float64)
        for p in wrist_positions
    ]
    n = len(positions)
    if n < 2:
        return 1

    run = 0
    for i in range(1, n):
        delta = float(np.linalg.norm(positions[i] - positions[i - 1]))
        if delta > threshold:
            run += 1
            if run >= consecutive:
                # Return the frame that *started* the run.
                return i - consecutive + 1
        else:
            run = 0
    return 1


def detect_jump_frame(
    hip_velocities: ArrayLike,
    knee_angles: ArrayLike,
    fps: float,
    search_start: int = 0,
) -> int:
    """
    Identifies the Load-and-Gather frame: the moment the bowler coils before
    delivery.  Detected as the knee-flexion minimum, confirmed by a local hip-
    velocity peak within a fps-scaled window around that dip.

    Parameters
    ----------
    hip_velocities : 1-D array-like
        Vertical hip velocity (signed, positive = upward).
    knee_angles : 1-D array-like
        Front-knee angle in degrees (smaller = more flexed).
    fps : float
        Frames per second — used to size the search window.
    search_start : int
        Ignore frames before this index (set to movement_start_frame).

    Returns
    -------
    int
        Frame index of the gather/jump.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    hip_v = _to_array(hip_velocities)
    knee_a = _to_array(knee_angles)

    n = min(len(hip_v), len(knee_a))
    if n == 0:
        return search_start

    # Only search after movement has started.
    search_start = max(0, min(search_start, n - 1))
    knee_roi = knee_a[search_start:]

    if len(knee_roi) == 0:
        return search_start

    knee_dip_rel = int(np.argmin(knee_roi))
    knee_dip = search_start + knee_dip_rel

    # Scale the confirmation window to fps (≈ 0.25 s either side).
    half_window = max(1, int(round(fps * 0.25)))
    lo = max(search_start, knee_dip - half_window)
    hi = min(n, knee_dip + half_window + 1)

    window = hip_v[lo:hi]
    if len(window) == 0:
        return knee_dip

    local_peak = int(np.argmax(window))
    jump_frame = lo + local_peak
    return int(jump_frame)


def detect_delivery_phase(
    wrist_positions: ArrayLike,
    wrist_velocities: ArrayLike,
    fps: float,
    search_start: int = 0,
) -> int:
    """
    Detects the DELIVERY phase: when the right hand (wrist) reaches its highest point
    (minimum Y coordinate) during the bowling action.
    
    This is more accurate than peak wrist velocity for detecting the actual release moment.
    
    Parameters
    ----------
    wrist_positions : list of (x, y) tuples or arrays
        Per-frame wrist position in pixels.
    wrist_velocities : 1-D array-like
        Wrist velocity per frame.
    fps : float
        Frames per second.
    search_start : int
        Only search for delivery at or after this frame.
    
    Returns
    -------
    int
        Frame index where wrist is at highest point (minimum Y).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    
    # Extract Y coordinates (height) from wrist positions
    wrist_y = []
    for pos in wrist_positions:
        if pos is None:
            wrist_y.append(None)
        else:
            pos_arr = np.asarray(pos, dtype=np.float64)
            if len(pos_arr) >= 2:
                wrist_y.append(pos_arr[1])  # Y coordinate
            else:
                wrist_y.append(None)
    
    wrist_y_arr = _to_array(wrist_y, fill_value=np.inf)
    n = len(wrist_y_arr)
    
    if n == 0:
        return search_start
    
    search_start = max(0, min(search_start, n - 1))
    
    # Find minimum Y (highest point) in the search region
    roi = wrist_y_arr[search_start:]
    if len(roi) == 0:
        return search_start
    
    # Find the frame with minimum Y (highest wrist position)
    min_y_rel = int(np.argmin(roi))
    delivery_frame = search_start + min_y_rel
    
    return int(delivery_frame)


# ---------------------------------------------------------------------------
# Top-level segmenter
# ---------------------------------------------------------------------------

def segment_phases(
    wrist_positions: ArrayLike,
    wrist_velocities: ArrayLike,
    hip_velocities: ArrayLike,
    knee_angles: ArrayLike,
    fps: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Segments a bowling delivery into four phases and returns per-frame labels.

    Phase boundaries (all frame indices, inclusive on the left):

        RUN-UP        [0, movement_start_frame)
        LOAD-GATHER   [movement_start_frame, jump_frame)
        DELIVERY      [jump_frame, peak_wrist_frame)
        FOLLOW-THROUGH [peak_wrist_frame, end)

    Parameters
    ----------
    wrist_positions : list of (2,) or (3,) array-like, length N
    wrist_velocities : 1-D array-like, length N
    hip_velocities : 1-D array-like, length N
    knee_angles : 1-D array-like, length N
    fps : float

    Returns
    -------
    labels : np.ndarray of object, shape (N,)
        Per-frame phase name.
    boundaries : dict[str, int]
        Boundary frame indices:
            movement_start_frame, jump_frame, peak_wrist_frame, release_frame
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    wrist_vel_arr = _to_array(wrist_velocities)
    hip_vel_arr   = _to_array(hip_velocities)
    knee_ang_arr  = _to_array(knee_angles)
    n_frames      = len(wrist_vel_arr)

    if n_frames == 0:
        raise ValueError("wrist_velocities is empty — no frames to segment.")

    # ----- Detect boundaries ------------------------------------------------
    movement_start = detect_movement_start(wrist_positions)
    jump_frame     = detect_jump_frame(
        hip_vel_arr, knee_ang_arr, fps, search_start=movement_start
    )
    peak_wrist     = detect_delivery_phase(
        wrist_positions, wrist_vel_arr, fps, search_start=jump_frame
    )

    # ----- Validate: boundaries must be strictly increasing -----------------
    def _clamped(val: int, lower: int, name: str) -> int:
        if val <= lower:
            warnings.warn(
                f"{name}={val} is not strictly greater than {lower}; "
                f"clamping to {lower + 1}.  Check your input signals.",
                UserWarning,
                stacklevel=2,
            )
            return lower + 1
        return val

    movement_start = max(1, min(movement_start, n_frames - 3))
    jump_frame     = _clamped(jump_frame, movement_start, "jump_frame")
    jump_frame     = min(jump_frame, n_frames - 2)
    peak_wrist     = _clamped(peak_wrist, jump_frame, "delivery_frame")
    peak_wrist     = min(peak_wrist, n_frames - 1)

    # ----- Build label array ------------------------------------------------
    labels = np.empty(n_frames, dtype=object)
    labels[:movement_start]           = "RUN-UP"
    labels[movement_start:jump_frame] = "LOAD-GATHER"
    labels[jump_frame:peak_wrist]     = "DELIVERY"
    labels[peak_wrist:]               = "FOLLOW-THROUGH"

    boundaries = {
        "movement_start_frame": movement_start,
        "jump_frame":           jump_frame,
        "peak_wrist_frame":     peak_wrist,
        "release_frame":        peak_wrist,   # alias for downstream consumers
    }

    return labels, boundaries
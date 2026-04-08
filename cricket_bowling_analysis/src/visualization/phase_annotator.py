"""
src/visualization/phase_annotator.py
---------------------------------------
Draws a colour-coded phase badge top-left of every frame.
On the release frame adds a red border flash + RELEASE text.

Phase colours:
    RUN-UP         — blue
    LOAD-UP        — yellow
    DELIVERY       — red-orange
    FOLLOW-THROUGH — green
"""

import cv2
import numpy as np

PHASE_COLORS = {
    "RUN-UP":          (255, 200, 0),
    "LOAD-UP":         (0, 200, 255),
    "DELIVERY":        (0, 80, 255),
    "FOLLOW-THROUGH":  (0, 200, 100),
}


def draw_phase(frame: np.ndarray, phase: str,
               is_release: bool = False,
               frame_idx: int = None) -> np.ndarray:
    """
    Draw phase badge and optional release flash.

    Returns:
        Annotated BGR frame (copy).
    """
    out   = frame.copy()
    color = PHASE_COLORS.get(phase, (200, 200, 200))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), _ = cv2.getTextSize(phase, font, 0.8, 2)
    pad = 10
    cv2.rectangle(out, (10, 10),
                  (10 + text_w + 2*pad, 10 + text_h + 2*pad),
                  color, -1)
    cv2.putText(out, phase, (10 + pad, 10 + text_h + pad),
                font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    if frame_idx is not None:
        cv2.putText(out, f"frame {frame_idx}",
                    (10, out.shape[0] - 10),
                    font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    if is_release:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (out.shape[1], out.shape[0]),
                      (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.rectangle(out, (0, 0),
                      (out.shape[1]-1, out.shape[0]-1), (0, 0, 255), 4)
        cv2.putText(out, "RELEASE",
                    (out.shape[1]//2 - 60, 50),
                    font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    return out
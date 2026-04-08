"""
src/visualization/ball_trail_drawer.py
----------------------------------------
Draws the ball and a fading 20-frame trail showing its path.
Use this to verify DeepSORT is tracking correctly.
Call reset() between deliveries to clear the trail.
"""

import cv2
import numpy as np
from collections import deque


class BallTrailDrawer:

    def __init__(self, trail_length: int = 20):
        self.trail        = deque(maxlen=trail_length)
        self.trail_length = trail_length

    def draw(self, frame: np.ndarray, ball_position,
             ball_color=(0, 165, 255),
             trail_color=(0, 100, 255)) -> np.ndarray:
        """
        Draw ball and trail on a frame.

        Args:
            ball_position : (cx, cy) or None.

        Returns:
            Annotated BGR frame (copy).
        """
        out = frame.copy()
        self.trail.append(ball_position)

        for i, pos in enumerate(self.trail):
            if pos is None:
                continue
            alpha  = (i + 1) / len(self.trail)
            radius = max(2, int(6 * alpha))
            color  = tuple(int(c * alpha) for c in trail_color)
            cv2.circle(out, (int(pos[0]), int(pos[1])),
                        radius, color, -1, cv2.LINE_AA)

        if ball_position is not None:
            cx, cy = int(ball_position[0]), int(ball_position[1])
            cv2.circle(out, (cx, cy), 8, ball_color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def reset(self):
        self.trail.clear()
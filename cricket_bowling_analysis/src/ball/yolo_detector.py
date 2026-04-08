"""
src/ball/yolo_detector.py
---------------------------
Runs YOLOv8 on every frame to detect the cricket ball.
Returns bounding box (cx, cy, w, h, conf) per frame or None.

Uses pretrained yolov8n.pt by default.
For better accuracy, fine-tune on your own cricket ball footage
and update yolo_model_path in config.yaml.
"""

from ultralytics import YOLO
from src.utils.config_loader import get_config


def detect_ball_sequence(frames: list) -> list:
    """
    Run YOLOv8 ball detection on all frames.

    Returns:
        List of dicts {"cx", "cy", "w", "h", "conf"} or None per frame.
    """
    cfg        = get_config()["ball"]
    model_path = cfg["yolo_model_path"]
    confidence = cfg["yolo_confidence"]

    try:
        model = YOLO(model_path)
    except Exception:
        print(f"[BALL] Model not found at {model_path}. Using pretrained yolov8n.pt")
        model = YOLO("yolov8n.pt")

    detections = []

    for i, frame in enumerate(frames):
        results = model(frame, verbose=False)[0]
        ball    = _extract_ball(results, confidence)
        detections.append(ball)

        if i % 100 == 0:
            found = sum(1 for d in detections if d is not None)
            print(f"[BALL] Frame {i}/{len(frames)} — detected in {found}/{i+1}")

    found = sum(1 for d in detections if d is not None)
    print(f"[BALL] Done. Ball detected in {found}/{len(frames)} frames.")
    return detections


def _extract_ball(results, min_confidence: float):
    """Extract highest-confidence ball bbox from one frame's results."""
    best      = None
    best_conf = min_confidence

    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])

        # class 32 = sports ball (COCO), class 0 if fine-tuned on cricket ball
        if cls in (0, 32) and conf > best_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = {
                "cx":   (x1 + x2) / 2,
                "cy":   (y1 + y2) / 2,
                "w":    x2 - x1,
                "h":    y2 - y1,
                "conf": conf,
            }
            best_conf = conf

    return best
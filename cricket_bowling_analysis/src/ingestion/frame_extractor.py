"""
src/ingestion/frame_extractor.py
-----------------------------------
Extracts frames from all 4 camera videos and applies sync offsets
so every camera starts at the same moment in time.
"""

import cv2


def extract_all_cameras(session: dict, sync_offsets: dict) -> dict:
    """
    Extract synced frames from all cameras (1 or 4).
    For single video mode, returns frames under the video's name.
    For multi-camera mode, returns frames for all 4 cameras.

    Args:
        session      : SessionConfig from session_loader.
        sync_offsets : {view: flash_frame_index} from flash_sync.

    Returns:
        {view: [np.ndarray, ...]} — all views same frame count.
    """
    # All cameras trim to the latest flash frame as the common start point
    start_frame = max(sync_offsets.values()) if sync_offsets else 0

    all_frames = {}
    for view, path in session["video_paths"].items():
        offset     = sync_offsets.get(view, 0)
        trim_start = start_frame - offset
        frames     = _read_frames(path, skip=trim_start)
        all_frames[view] = frames
        print(f"[OK] {view}: {len(frames)} frames after sync (offset={offset})")

    # Truncate all to shortest so every view is same frame count
    min_len = min(len(f) for f in all_frames.values())
    for view in all_frames:
        all_frames[view] = all_frames[view][:min_len]

    mode_str = "single video" if session.get("single_video_mode") else "4-camera"
    print(f"[OK] All cameras ({mode_str}): {min_len} frames")
    return all_frames


def _read_frames(path: str, skip: int = 0) -> list:
    cap    = cv2.VideoCapture(path)
    frames = []
    idx    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= skip:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames
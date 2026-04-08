"""
src/sync/frame_aligner.py
---------------------------
Validates flash offsets and reports alignment quality.
Actual frame trimming is done in frame_extractor.py using these offsets.
"""


def validate_alignment(offsets: dict) -> dict:
    """
    Validate sync offsets and compute alignment quality.

    Returns:
        {
            offsets         : original offsets
            common_start    : frame all cameras trim to
            max_drift_frames: max difference between any two cameras
            is_reliable     : True if drift < 3 frames
        }
    """
    common_start = max(offsets.values())
    min_offset   = min(offsets.values())
    max_drift    = common_start - min_offset
    is_reliable  = max_drift <= 3

    if is_reliable:
        print(f"[SYNC] Alignment OK — max drift: {max_drift} frames")
    else:
        print(f"[WARNING] Poor sync — max drift: {max_drift} frames. "
              f"Check LED flash is visible from all cameras.")

    return {
        "offsets":          offsets,
        "common_start":     common_start,
        "max_drift_frames": max_drift,
        "is_reliable":      is_reliable,
    }
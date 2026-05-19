"""
src/sync/__init__.py
────────────────────
Manual frame-based synchronization for multi-camera setup.
"""

from .manual_sync import (
    manual_sync_interactive,
    manual_sync_from_file,
    save_sync_offsets,
    get_manual_sync_frames,
    calculate_manual_offsets,
    validate_manual_offsets,
    display_sync_summary,
)

__all__ = [
    "manual_sync_interactive",
    "manual_sync_from_file",
    "save_sync_offsets",
    "get_manual_sync_frames",
    "calculate_manual_offsets",
    "validate_manual_offsets",
    "display_sync_summary",
]

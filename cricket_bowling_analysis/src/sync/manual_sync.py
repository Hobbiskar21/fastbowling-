"""
src/sync/manual_sync.py
─────────────────────────────────────────────────────────────────────────────
Manual frame-based synchronization for multi-camera setup.

Instead of auto-detecting flash, user manually specifies frame numbers
where a key event occurs in each camera.

WORKFLOW:
    1. Back camera starts from frame 0 (reference)
    2. Ask user: "At what frame does the event occur in BACK camera?"
    3. Ask user: "At what frame does the SAME event occur in SIDE camera?"
    4. Ask user: "At what frame does the SAME event occur in FRONT camera?"
    5. Calculate offsets based on these frame numbers
    6. Extract frames with offsets applied

EXAMPLE:
    Back camera:  event at frame 0 (reference)
    Side camera:  event at frame 15 → offset = 0 - 15 = -15
    Front camera: event at frame 8  → offset = 0 - 8 = -8

    After sync:
    Back frame 0 = Side frame 15 = Front frame 8 (same moment!)
"""

from typing import Dict, Optional, Tuple
import sys


def get_manual_sync_frames() -> Dict[str, int]:
    """
    Interactively ask user for cascading sync frame numbers.
    
    Workflow:
        1. Back camera starts at frame 0
        2. Ask: "At which frame in BACK does SIDE-ON start?"
        3. Ask: "At which frame in SIDE-ON video do you want to sync?"
        4. Ask: "At which frame in FRONT does SIDE-ON end?"
        5. Ask: "At which frame in FRONT video do you want to sync?"
    
    Returns
    -------
    dict
        {
            "back": 0,           # back camera reference
            "side_start": 20,    # frame in back where side-on starts
            "side_sync": 3,      # frame in side video to sync
            "front_start": 50,   # frame in back where front-on starts
            "front_sync": 5,     # frame in front video to sync
        }
    """
    print("\n" + "="*70)
    print("CASCADING MANUAL FRAME SYNCHRONIZATION")
    print("="*70)
    print("\nBack camera starts at frame 0 (reference)")
    print("Then specify where SIDE-ON and FRONT-ON cameras should sync.\n")

    sync_frames = {}

    # Back camera reference
    sync_frames["back"] = 0
    print("-" * 70)
    print("BACK CAMERA (Reference)")
    print("-" * 70)
    print("✓ Back camera starts at frame 0")

    # Side camera sync point
    print("\n" + "-" * 70)
    print("SIDE-ON CAMERA SYNC")
    print("-" * 70)
    while True:
        try:
            side_start = int(input("At which frame in BACK camera does SIDE-ON view start? ").strip())
            if side_start < 0:
                print("[ERROR] Frame number must be >= 0")
                continue
            sync_frames["side_start"] = side_start
            print(f"✓ Side-on starts at back frame {side_start}")
            break
        except ValueError:
            print("[ERROR] Please enter a valid integer")

    while True:
        try:
            side_sync = int(input("At which frame in SIDE-ON video do you want to sync? ").strip())
            if side_sync < 0:
                print("[ERROR] Frame number must be >= 0")
                continue
            sync_frames["side_sync"] = side_sync
            print(f"✓ Side-on sync frame: {side_sync}")
            print(f"  → Back frame {side_start} = Side frame {side_sync}")
            break
        except ValueError:
            print("[ERROR] Please enter a valid integer")

    # Front camera sync point
    print("\n" + "-" * 70)
    print("FRONT-ON CAMERA SYNC")
    print("-" * 70)
    while True:
        try:
            front_start = int(input("At which frame in BACK camera does FRONT-ON view start? ").strip())
            if front_start < 0:
                print("[ERROR] Frame number must be >= 0")
                continue
            sync_frames["front_start"] = front_start
            print(f"✓ Front-on starts at back frame {front_start}")
            break
        except ValueError:
            print("[ERROR] Please enter a valid integer")

    while True:
        try:
            front_sync = int(input("At which frame in FRONT-ON video do you want to sync? ").strip())
            if front_sync < 0:
                print("[ERROR] Frame number must be >= 0")
                continue
            sync_frames["front_sync"] = front_sync
            print(f"✓ Front-on sync frame: {front_sync}")
            print(f"  → Back frame {front_start} = Front frame {front_sync}")
            break
        except ValueError:
            print("[ERROR] Please enter a valid integer")

    return sync_frames


def calculate_manual_offsets(sync_frames: Dict[str, int]) -> Dict[str, int]:
    """
    Calculate frame offsets from cascading sync points.
    
    Parameters
    ----------
    sync_frames : dict
        {
            "back": 0,           # back camera reference
            "side_start": 20,    # frame in back where side-on starts
            "side_sync": 3,      # frame in side video to sync
            "front_start": 50,   # frame in back where front-on starts
            "front_sync": 5,     # frame in front video to sync
        }
    
    Returns
    -------
    dict
        {
            "back": 0,      # reference
            "side": 17,     # offset for side camera
            "front": 45,    # offset for front camera
        }
    
    Calculation:
        side_offset = side_start - side_sync
        front_offset = front_start - front_sync
        
    Example:
        side_start = 20, side_sync = 3
        side_offset = 20 - 3 = 17
        
        This means: skip first 17 frames of side camera
        So side frame 17 aligns with back frame 20
    """
    back_ref = sync_frames["back"]
    side_start = sync_frames["side_start"]
    side_sync = sync_frames["side_sync"]
    front_start = sync_frames["front_start"]
    front_sync = sync_frames["front_sync"]
    
    # Calculate offsets
    side_offset = side_start - side_sync
    front_offset = front_start - front_sync
    
    offsets = {
        "back": back_ref,
        "side": side_offset,
        "front": front_offset,
    }
    
    return offsets


def validate_manual_offsets(offsets: Dict[str, int], max_drift: int = 50) -> Tuple[bool, str]:
    """
    Validate that offsets are reasonable.
    
    Parameters
    ----------
    offsets : dict
        Offsets from calculate_manual_offsets
    max_drift : int
        Maximum acceptable drift between cameras (frames)
    
    Returns
    -------
    tuple
        (is_valid, message)
    """
    offset_values = [abs(o) for o in offsets.values()]
    max_offset = max(offset_values)
    
    if max_offset > max_drift:
        return False, f"Max drift {max_offset} exceeds threshold {max_drift}"
    
    return True, f"Offsets valid (max drift: {max_offset} frames)"


def display_sync_summary(sync_frames: Dict[str, int], offsets: Dict[str, int]) -> None:
    """
    Display a summary of the cascading synchronization.
    
    Parameters
    ----------
    sync_frames : dict
        Frame numbers from user input
    offsets : dict
        Calculated offsets
    """
    print("\n" + "="*70)
    print("SYNCHRONIZATION SUMMARY")
    print("="*70)
    
    print("\nCascading Sync Points:")
    print(f"  Back camera:     starts at frame {sync_frames['back']}")
    print(f"  Side-on starts:  at back frame {sync_frames['side_start']}")
    print(f"  Side-on sync:    at side frame {sync_frames['side_sync']}")
    print(f"  Front-on starts: at back frame {sync_frames['front_start']}")
    print(f"  Front-on sync:   at front frame {sync_frames['front_sync']}")
    
    print("\nCalculated Offsets (frames to skip):")
    print(f"  Back:   {offsets['back']:+d} (reference)")
    print(f"  Side:   {offsets['side']:+d} (skip first {offsets['side']} frames)")
    print(f"  Front:  {offsets['front']:+d} (skip first {offsets['front']} frames)")
    
    print("\nAlignment:")
    print(f"  Back frame {sync_frames['side_start']} = Side frame {sync_frames['side_sync']}")
    print(f"  Back frame {sync_frames['front_start']} = Front frame {sync_frames['front_sync']}")
    
    print("\nAfter Synchronization:")
    print(f"  Back frame 0 = Side frame {offsets['side']} = Front frame {offsets['front']}")
    print("  (All three cameras start at the same moment)")
    print("="*70 + "\n")


def manual_sync_interactive() -> Dict[str, int]:
    """
    Full interactive manual sync workflow.
    
    Returns
    -------
    dict
        {
            "back": 0,
            "side": -15,
            "front": -8,
        }
    """
    # Get frame numbers from user
    sync_frames = get_manual_sync_frames()
    
    # Calculate offsets
    offsets = calculate_manual_offsets(sync_frames)
    
    # Validate
    is_valid, msg = validate_manual_offsets(offsets)
    print(f"\n[SYNC] {msg}")
    
    if not is_valid:
        print("[ERROR] Offsets are too large. Please re-enter frame numbers.")
        return manual_sync_interactive()  # Retry
    
    # Display summary
    display_sync_summary(sync_frames, offsets)
    
    # Confirm
    while True:
        confirm = input("Proceed with these offsets? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            print("[SYNC] ✓ Manual sync confirmed")
            return offsets
        elif confirm in ["no", "n"]:
            print("[SYNC] Restarting manual sync...")
            return manual_sync_interactive()  # Restart
        else:
            print("[ERROR] Please enter 'yes' or 'no'")


def manual_sync_from_file(sync_file: str) -> Dict[str, int]:
    """
    Load manual sync offsets from a file.
    
    File format (sync.txt):
    ```
    back: 0
    side: -15
    front: -8
    ```
    
    Parameters
    ----------
    sync_file : str
        Path to sync file
    
    Returns
    -------
    dict
        Offsets dictionary
    """
    offsets = {}
    
    try:
        with open(sync_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                
                camera = parts[0].strip().lower()
                offset = int(parts[1].strip())
                offsets[camera] = offset
        
        print(f"[SYNC] Loaded manual offsets from {sync_file}")
        print(f"[SYNC] Offsets: {offsets}")
        return offsets
    
    except FileNotFoundError:
        print(f"[ERROR] Sync file not found: {sync_file}")
        return None
    except ValueError as e:
        print(f"[ERROR] Invalid sync file format: {e}")
        return None


def save_sync_offsets(offsets: Dict[str, int], output_file: str) -> None:
    """
    Save manual sync offsets to a file for future use.
    
    Parameters
    ----------
    offsets : dict
        Offsets dictionary
    output_file : str
        Path to output file
    """
    try:
        with open(output_file, "w") as f:
            f.write("# Manual Sync Offsets\n")
            f.write("# Format: camera: offset\n")
            f.write("# Positive offset = camera is behind (skip first N frames)\n")
            f.write("# Negative offset = camera is ahead (start from frame N)\n\n")
            
            for camera, offset in offsets.items():
                f.write(f"{camera}: {offset}\n")
        
        print(f"[SYNC] Saved offsets to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save offsets: {e}")

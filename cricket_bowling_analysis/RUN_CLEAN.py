#!/usr/bin/env python3
"""
Clean runner - clears all caches and runs fresh.
Use this instead of main.py or run_single_video.py to ensure no cached code.

USAGE:
    python RUN_CLEAN.py --session data/raw/sessions/session_001
    (for multi-camera with manual sync)
    
    OR
    
    python RUN_CLEAN.py --video input_videos/single/bowling_video1.mp4
    (for single video)
"""

import sys
import os
import shutil

# STEP 0: FORCE PYTHON TO NOT USE BYTECODE
sys.dont_write_bytecode = True

# STEP 1: Clear all pycache BEFORE importing anything
print("=" * 70)
print("CLEARING ALL PYTHON CACHE...")
print("=" * 70)

for root, dirs, files in os.walk("."):
    if "__pycache__" in dirs:
        pycache_path = os.path.join(root, "__pycache__")
        print(f"Removing: {pycache_path}")
        shutil.rmtree(pycache_path, ignore_errors=True)
    
    for file in files:
        if file.endswith(".pyc"):
            pyc_path = os.path.join(root, file)
            print(f"Removing: {pyc_path}")
            os.remove(pyc_path)

print("\nCache cleared!\n")

# STEP 2: Run fresh
print("=" * 70)
print("RUNNING WITH FRESH CODE")
print("=" * 70)
print()

import argparse

parser = argparse.ArgumentParser(description="Clean Runner - Multi-Camera or Single Video")
parser.add_argument("--session", default=None, help="Path to session folder (multi-camera mode)")
parser.add_argument("--video", default=None, help="Path to .mp4 file (single video mode)")
parser.add_argument("--camera", default="side", help="Which camera to analyze (multi-camera mode)")
parser.add_argument("--output", default="outputs", help="Output directory")
parser.add_argument("--sync-file", default=None, help="Optional path to pre-saved sync offsets file")
args = parser.parse_args()

# Determine mode
if args.session:
    # Multi-camera mode
    print("MODE: Multi-Camera with Manual Sync")
    print("=" * 70)
    
    from main import run_pipeline
    
    if not os.path.exists(args.session):
        print(f"[ERROR] Session not found: {args.session}")
        sys.exit(1)
    
    print(f"\nSession: {args.session}")
    print(f"Camera: {args.camera}")
    print(f"Output: {args.output}")
    if args.sync_file:
        print(f"Sync file: {args.sync_file}")
    print()
    
    run_pipeline(
        session_path=args.session,
        camera=args.camera,
        output_dir=args.output,
        sync_file=args.sync_file,
    )

elif args.video:
    # Single video mode
    print("MODE: Single Video")
    print("=" * 70)
    
    from run_single_video import run_single_video
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)
    
    print(f"\nVideo: {args.video}")
    print(f"Output: {args.output}")
    print()
    
    run_single_video(args.video, args.output)

else:
    # Interactive single video mode
    print("MODE: Single Video (Interactive)")
    print("=" * 70)
    print()
    
    from run_single_video import run_single_video, select_video_interactive
    
    video_path = select_video_interactive()
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)
    
    run_single_video(video_path, args.output)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

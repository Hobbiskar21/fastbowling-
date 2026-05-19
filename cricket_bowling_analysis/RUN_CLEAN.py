#!/usr/bin/env python3
"""
Clean runner for three-angle / multi-camera analysis.
Use this instead of calling main.py directly when you want a fresh run.

USAGE:
    python RUN_CLEAN.py --session data/raw/sessions/session_001
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

parser = argparse.ArgumentParser(description="Clean Runner - Three-angle / multi-camera analysis")
parser.add_argument("--session", required=True, help="Path to session folder")
parser.add_argument("--camera", default="side", help="Which camera to analyze")
parser.add_argument("--output", default="outputs", help="Output directory")
parser.add_argument("--sync-file", default=None, help="Optional path to pre-saved sync offsets file")
args = parser.parse_args()

print("MODE: Three-Angle / Multi-Camera")
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

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

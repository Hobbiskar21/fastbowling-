"""
src/storage/csv_writer.py
---------------------------
Manages CSV storage for bowling analysis.

For single video mode:
    - Uses a single CSV file: data/processed/single_video_deliveries.csv
    - Checks if video already exists and asks for overwrite confirmation
    - Replaces existing data for that video in-place

For multi-camera/session mode:
    - Uses per-session CSV: data/processed/sessions/{session_id}/results/deliveries.csv
"""

import os
import csv
import pandas as pd
from src.utils.config_loader import get_config


def _get_workspace_root() -> str:
    storage_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(storage_dir)
    analysis_dir = os.path.dirname(src_dir)
    project_dir = os.path.dirname(analysis_dir)
    return os.path.dirname(project_dir)


def save_delivery(record: dict, session_id: str, single_video_mode: bool = False) -> str:
    """
    Save a DeliveryRecord to CSV.
    
    For single_video_mode:
        - Uses single shared CSV file
        - Checks for existing data and asks for overwrite confirmation
        - Replaces data in-place if confirmed
    
    For session mode:
        - Uses per-session CSV file
        - Appends new deliveries
    
    Returns:
        Path to the CSV file.
    """
    cfg = get_config()
    
    if single_video_mode:
        # Single video mode: use shared CSV file
        csv_path = os.path.join(_get_workspace_root(), "data", "processed", "single_video_deliveries.csv")
        print(f"[STORAGE] CSV path: {csv_path}")
        
        # Check if video already exists in CSV (only if file exists)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Check if this session_id already exists
                if session_id in df["session_id"].values:
                    print(f"\n[WARNING] Video '{session_id}' already exists in CSV")
                    response = input("Overwrite existing data? (yes/no): ").strip().lower()
                    if response == "yes":
                        # Remove existing rows for this session
                        df = df[df["session_id"] != session_id]
                        df.to_csv(csv_path, index=False)
                        print(f"[INFO] Removed existing data for '{session_id}'")
                    else:
                        print(f"[INFO] Keeping existing data for '{session_id}'")
                        return csv_path
            except Exception as e:
                print(f"[WARNING] Could not read existing CSV: {e}")
                print(f"[INFO] Will create new CSV file")
        else:
            print(f"[STORAGE] CSV file does not exist yet, will create new one")
        
        # Append new record
        record_with_nulls = {
            key: "null" if value is None else value
            for key, value in record.items()
        }
        
        file_exists = os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(f"[STORAGE] Directory created/verified: {os.path.dirname(csv_path)}")
        
        try:
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record_with_nulls.keys())
                if not file_exists:
                    writer.writeheader()
                    print(f"[STORAGE] CSV header written")
                writer.writerow(record_with_nulls)
                print(f"[STORAGE] CSV row written")
        except Exception as e:
            print(f"[ERROR] Failed to write CSV: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print(f"[STORAGE] Delivery saved to shared CSV: {csv_path}")
    
    else:
        # Session mode: use per-session CSV file
        output_dir = os.path.join(
            cfg["paths"]["processed_sessions"],
            session_id,
            "results"
        )
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "deliveries.csv")
        file_exists = os.path.exists(csv_path)
        
        # Convert None values to "null" string
        record_with_nulls = {
            key: "null" if value is None else value
            for key, value in record.items()
        }
        
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record_with_nulls.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record_with_nulls)
        
        print(f"[STORAGE] Delivery {record.get('delivery_number')} "
              f"saved to {csv_path}")
    
    return csv_path

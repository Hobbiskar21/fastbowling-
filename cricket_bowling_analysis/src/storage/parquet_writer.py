"""
src/storage/parquet_writer.py
-------------------------------
FUTURE — activate when adding ML training.
Change storage.backend in config.yaml to "parquet" to use this.

Same save_delivery() interface as csv_writer — one import line change.
Files partitioned by bowler_id/year/session_id for fast ML queries.

Install when ready: pip install pyarrow pandas
"""

import os
import pandas as pd


def save_delivery(record: dict, session_id: str,
                  bowler_id: str = "unknown", year: str = "2025") -> str:
    """
    Save a DeliveryRecord as a Parquet file.

    Returns:
        Path to the Parquet file.
    """
    output_dir = os.path.join(
        "data", "parquet", "biomechanics",
        f"bowler_id={bowler_id}",
        f"year={year}",
        f"session_id={session_id}",
    )
    os.makedirs(output_dir, exist_ok=True)

    delivery_num = record.get("delivery_number", 0)
    parquet_path = os.path.join(output_dir, f"delivery_{delivery_num:03d}.parquet")

    df = pd.DataFrame([record])
    df.to_parquet(parquet_path, index=False)

    print(f"[STORAGE] Delivery {delivery_num} saved to {parquet_path}")
    return parquet_path


def load_session_deliveries(session_id: str, bowler_id: str = "*",
                             year: str = "*") -> "pd.DataFrame":
    """Load all deliveries for a session into a DataFrame for ML training."""
    import glob
    pattern = os.path.join(
        "data", "parquet", "biomechanics",
        f"bowler_id={bowler_id}", f"year={year}",
        f"session_id={session_id}", "*.parquet"
    )
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
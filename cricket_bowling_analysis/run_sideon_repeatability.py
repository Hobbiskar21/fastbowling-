"""Root runner for automatic side-on repeatability preprocessing and scoring."""

import argparse
from pathlib import Path

from src.repeatability.config import DEFAULT_OUTPUT_DIR
from src.repeatability.pipeline import create_output_dirs, process_csv
from src.repeatability.repeatability_scorer import calculate_repeatability_scores
from src.repeatability.repeatability_visualizer import plot_all_key_features


def main():
    parser = argparse.ArgumentParser(description="Automatic side-on repeatability pipeline from frame-wise CSVs.")
    parser.add_argument("--input_csvs", nargs="+", required=True, help="Frame-wise feature CSV paths.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dirs = create_output_dirs(args.output_dir)
    curve_paths = []
    for csv_path in args.input_csvs:
        if not Path(csv_path).exists():
            print(f"[REPEATABILITY] Missing input CSV, skipping: {csv_path}")
            continue
        curve_paths.append(str(process_csv(csv_path, dirs)["curves"]))

    if curve_paths:
        plot_all_key_features(curve_paths, str(dirs["graphs"]))
        calculate_repeatability_scores(curve_paths, str(dirs["scores"] / "repeatability_scores.csv"))
    print(f"[REPEATABILITY] Complete. Output root: {args.output_dir}")


if __name__ == "__main__":
    main()

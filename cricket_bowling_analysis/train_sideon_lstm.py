"""Root runner for training the side-on repeatability LSTM."""

import argparse
from pathlib import Path

import pandas as pd

from src.repeatability.lstm_dataset_builder import build_lstm_dataset

try:
    from src.repeatability.train_lstm import train_lstm_model
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "[REPEATABILITY] PyTorch is not installed in this Python environment.\n"
            "Install project dependencies first:\n"
            "  python -m pip install -r requirements.txt\n"
            "Then rerun:\n"
            "  python train_sideon_lstm.py --epochs 30"
        ) from exc
    raise


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return workspace_root() / candidate


def metadata_path(sequence_path: Path) -> Path:
    return sequence_path.with_name(sequence_path.name.replace("_sequence.npy", "_metadata.csv"))


def create_default_labels(sequence_paths: list[Path], labels_path: Path) -> Path:
    """Create simple positive labels from generated sequence metadata."""
    rows = []
    for sequence_path in sequence_paths:
        meta_path = metadata_path(sequence_path)
        if meta_path.exists():
            meta = pd.read_csv(meta_path).iloc[0].to_dict()
            bowler_id = str(meta.get("bowler_id", "unknown_bowler"))
            delivery_id = str(meta.get("delivery_id", sequence_path.stem.replace("_sequence", "")))
        else:
            bowler_id = "unknown_bowler"
            delivery_id = sequence_path.stem.replace("_sequence", "")
        rows.append({
            "bowler_id": bowler_id,
            "delivery_id": delivery_id,
            "label": 1,
            "split": "train",
        })

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    print(f"[REPEATABILITY] Created default labels from {len(rows)} sequence(s): {labels_path}")
    print("[REPEATABILITY] Default label=1 means these videos are treated as good/reference repeatability examples.")
    return labels_path


def main():
    parser = argparse.ArgumentParser(description="Train side-on repeatability LSTM.")
    parser.add_argument("--sequence_dir", default="outputs/repeatability/delivery_sequences")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--output_dir", default="outputs/repeatability/lstm_dataset")
    parser.add_argument("--model_output", default="outputs/repeatability/models/sideon_lstm.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    sequence_dir = resolve_path(args.sequence_dir)
    output_dir = resolve_path(args.output_dir)
    model_output = resolve_path(args.model_output)
    labels_csv = resolve_path(args.labels_csv) if args.labels_csv else output_dir / "auto_labels.csv"

    sequence_paths = sorted(sequence_dir.glob("*_sequence.npy"))
    if not sequence_paths:
        raise FileNotFoundError(
            f"No sequence NPY files found in {sequence_dir}\n"
            "Run run_sideon_repeatability_videos.py first to generate training sequences."
        )
    if args.labels_csv is None or not labels_csv.exists():
        create_default_labels(sequence_paths, labels_csv)

    print(f"[REPEATABILITY] Training on {len(sequence_paths)} sequence(s) from: {sequence_dir}")
    build_lstm_dataset([str(path) for path in sequence_paths], labels_csv, output_dir)
    train_lstm_model(output_dir, model_output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print(f"[REPEATABILITY] Model saved: {model_output}")


if __name__ == "__main__":
    main()

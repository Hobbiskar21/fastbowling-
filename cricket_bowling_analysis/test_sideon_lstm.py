"""Root runner for testing one delivery CSV with a trained side-on LSTM."""

import argparse

from src.repeatability.test_lstm_video import test_single_delivery


def main():
    parser = argparse.ArgumentParser(description="Test side-on LSTM on one frame-wise CSV.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="outputs/repeatability")
    args = parser.parse_args()

    result = test_single_delivery(
        args.input_csv,
        args.model_path,
        args.output_dir,
        review_phases=False,
    )
    print(f"[REPEATABILITY] Prediction: {result}")


if __name__ == "__main__":
    main()

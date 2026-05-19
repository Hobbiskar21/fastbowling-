# Fast Bowling Analysis

Simple tools for cricket fast-bowling video analysis.

This repo has two main workflows:

1. **Side-on repeatability**
2. **Three-angle sync / multi-camera analysis**

Videos, generated outputs, model weights, CSVs, graphs, and dashboards are ignored by Git. Keep private data in `input_videos/` and generated files in `outputs/`.

## Setup

```bat
cd cricket_bowling_analysis
python -m pip install -r requirements.txt
```

Put YOLO pose weights such as `yolov8m-pose.pt` in the working folder when running locally. Model weights are not committed.

## 1. Side-On Repeatability

Use this when you have side-on bowling videos and want a repeatability score.

### Put Training Videos

```text
input_videos/repeatability/train/
```

### Generate Training Data

```bat
python run_sideon_repeatability_videos.py --split train
```

Use this only when you want to rebuild everything:

```bat
python run_sideon_repeatability_videos.py --split train --force_analysis
```

The script skips videos that are already processed and removes artifacts for videos deleted from the training folder.

### Train LSTM

```bat
python train_sideon_lstm.py --epochs 30
```

This trains from:

```text
outputs/repeatability/delivery_sequences/
```

and saves the model to:

```text
outputs/repeatability/models/sideon_lstm.pt
```

### Score One Video

Interactive picker:

```bat
python score_sideon_repeatability_video.py
```

Direct path:

```bat
python score_sideon_repeatability_video.py C:\path\to\video.mp4
```

The scorer outputs:

- Overall repeatability score out of 100
- Verdict
- Strongest phase
- Weakest phase
- Seven phase scores
- Dashboard image
- Prediction CSV
- LSTM sequence graph

## 2. Three-Angle Sync / Multi-Camera

Use this when you have multiple camera angles for the same bowling action.

Typical folder shape:

```text
input_videos/multiple/<session_name>/
```

Run the multi-camera pipeline:

```bat
cd cricket_bowling_analysis
python main.py --session path\to\session_folder
```

For the cleaner wrapper:

```bat
cd cricket_bowling_analysis
python RUN_CLEAN.py --session path\to\session_folder
```

The sync step asks for the matching frame numbers across cameras. Outputs are written under `outputs/`.

## Notes

- Do not commit videos, model weights, generated CSVs, dashboards, or outputs.
- If results look stale, rerun the workflow with `--force_analysis`.
- If PyTorch is missing, install dependencies with `python -m pip install -r requirements.txt`.

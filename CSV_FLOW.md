# CSV Flow

## Single Video Output

Running `python RUN_CLEAN.py` or `python cricket_bowling_analysis/run_single_video.py` processes one video from `input_videos/single`.

The pipeline now writes two CSV outputs:

1. Delivery summary CSV
   - Written by `src/storage/csv_writer.py`
   - Contains one row for the delivery summary
   - Includes release frame, phase ranges, key biomechanical angles, velocities, bowling style, and run-up metrics

2. Frame-by-frame CSV
   - Written by `run_single_video.py`
   - Saved as `outputs/<session_id>_frame_data.csv`
   - Contains one row per frame
   - Includes phase label, release-frame marker, wrist/hip velocities, raw YOLO keypoints, smoothed YOLO keypoints, and frame angles

## Video Output

The annotated video is rendered through `src/utils/video_utils.py`.

The writer keeps the original frame size, applies the configured slowdown, normalizes frames before writing, and verifies that the output file was created with frames.

Expected single-video outputs:

```text
outputs/
+-- videos/
|   +-- single_<video_name>_analysis.mp4
+-- artifacts/
    +-- single_<video_name>_frame_data.csv
    +-- single_<video_name>_phase_report.json
    +-- single_<video_name>_runup_analysis.png
```

The shared delivery-summary CSV is stored separately:

```text
data/
+-- processed/
    +-- single_video_deliveries.csv
```

# CSV Creation and Storage Flow

## Complete End-to-End Flow

```
INPUT VIDEO
    ↓
    └─→ input_videos/bowling_video1.mp4
    
run_single_video.py
    ↓
    ├─ Creates session folder: data/raw/sessions/session_bowling_video1/
    ├─ Copies video there: data/raw/sessions/session_bowling_video1/bowling_video1.mp4
    └─ Calls run_pipeline(session_path, single_video=True)
    
main.py → run_pipeline()
    ↓
    ├─ Step 1: load_session(session_path)
    │   └─ Extracts session_id = "session_bowling_video1" (from folder basename)
    │
    ├─ Step 2-6: Process video
    │   ├─ Sync cameras
    │   ├─ Extract frames
    │   ├─ Detect pose (MediaPipe)
    │   ├─ Ball tracking (optional)
    │   └─ Compute biomechanics
    │
    ├─ Step 7: build_delivery_record()
    │   └─ Creates dict with all metrics:
    │       {
    │           "session_id": "session_bowling_video1",
    │           "delivery_number": 1,
    │           "release_frame": 288,
    │           "elbow_angle_at_release": 179.76,
    │           "shoulder_angle_max": 179.93,
    │           "arm_velocity_max": 1793.12,
    │           "runup_speed_mean": 34.64,
    │           "bowling_style": "FRONT_ON",
    │           ... (30+ fields total)
    │       }
    │
    ├─ Step 8: save_delivery(record, session_id)
    │   └─ Calls csv_writer.save_delivery()
    │
    └─ CSV WRITER (src/storage/csv_writer.py)
        ↓
        ├─ Reads config: cfg["paths"]["processed_sessions"] = "data/processed/sessions"
        │
        ├─ Constructs output directory:
        │   output_dir = os.path.join(
        │       "data/processed/sessions",      ← from config
        │       "session_bowling_video1",       ← session_id
        │       "results"                       ← hardcoded
        │   )
        │   = "data/processed/sessions/session_bowling_video1/results"
        │
        ├─ Creates directory if not exists:
        │   os.makedirs(output_dir, exist_ok=True)
        │
        ├─ Constructs CSV path:
        │   csv_path = "data/processed/sessions/session_bowling_video1/results/deliveries.csv"
        │
        ├─ Checks if file exists:
        │   file_exists = os.path.exists(csv_path)
        │
        ├─ Converts None values to "null" strings:
        │   record_with_nulls = {
        │       "release_frame": 288,
        │       "elbow_angle_at_release": 179.76,
        │       "missing_field": "null",  ← None becomes "null"
        │       ...
        │   }
        │
        ├─ Opens CSV file in append mode:
        │   with open(csv_path, "a", newline="") as f:
        │
        ├─ If first delivery (file doesn't exist):
        │   ├─ Writes header row with all field names
        │   └─ Example header:
        │       session_id,delivery_number,bowler_name,release_frame,elbow_angle_at_release,...
        │
        ├─ Writes data row:
        │   session_bowling_video1,1,unknown,288,179.76,...
        │
        └─ Prints confirmation:
            [STORAGE] Delivery 1 saved to data/processed/sessions\session_bowling_video1\results\deliveries.csv
```

## Key Points

### 1. **Session ID Extraction**
- Source: `session_path` basename
- Example: `data/raw/sessions/session_bowling_video1/` → `session_bowling_video1`
- Used as: Folder name in processed_sessions

### 2. **Config-Driven Paths**
```yaml
# config/config.yaml
paths:
  processed_sessions: "data/processed/sessions"  ← CSV root directory
```

### 3. **Directory Structure Created**
```
data/processed/sessions/
└── session_bowling_video1/          ← session_id
    └── results/                     ← hardcoded "results" folder
        └── deliveries.csv           ← hardcoded filename
```

### 4. **CSV File Format**
- **First row:** Column headers (all field names from record dict)
- **Subsequent rows:** Data values, one row per delivery
- **Append mode:** Multiple deliveries can be added to same CSV
- **Null handling:** Python `None` → CSV `"null"` string

### 5. **Data Fields in CSV** (30+ columns)
```
Identity:
  - session_id
  - delivery_number
  - bowler_name

Release Info:
  - release_frame
  - release_height_px
  - release_angle_deg

Angles at Release:
  - elbow_angle_at_release
  - shoulder_angle_at_release
  - front_knee_angle_at_release
  - back_knee_angle_at_release
  - hip_angle_at_release
  - trunk_lean_at_release
  - hip_shoulder_sep_at_release

Peak Angles (full delivery):
  - elbow_angle_max
  - shoulder_angle_max
  - front_knee_angle_min
  - hip_shoulder_sep_max
  - trunk_lean_max

Velocities:
  - arm_velocity_max
  - arm_velocity_mean
  - runup_speed_mean

Ball:
  - ball_speed_ms

Bowling Style:
  - bowling_style
  - bowling_style_front_score
  - bowling_style_side_score

Phase Ranges:
  - phase_runup_start/end
  - phase_loadup_start/end
  - phase_delivery_start/end
  - phase_followthrough_start/end
```

## Example CSV Output

```csv
session_id,delivery_number,bowler_name,release_frame,elbow_angle_at_release,shoulder_angle_max,arm_velocity_max,runup_speed_mean,bowling_style,bowling_style_front_score,bowling_style_side_score,...
session_bowling_video1,1,unknown,288,179.76,179.93,1793.12,34.64,FRONT_ON,0.75,0.25,...
session_bowling_video2,1,unknown,null,178.44,179.85,2246.39,298.78,SIDE_ON,0.20,0.80,...
```

## File Locations After Run

```
Input:
  input_videos/bowling_video1.mp4

Temporary (cleaned up):
  data/raw/sessions/session_bowling_video1/bowling_video1.mp4

Output (permanent):
  data/processed/sessions/session_bowling_video1/results/deliveries.csv
  outputs/single_video/session_bowling_video1_bowling_video1_analysis.avi
```

## Summary

1. **Video input** → Session folder created with video name
2. **Pipeline processes** → Extracts biomechanics metrics
3. **Record built** → Dict with 30+ fields including bowling style
4. **CSV saved** → Appended to `data/processed/sessions/{session_id}/results/deliveries.csv`
5. **Multiple deliveries** → Can append multiple rows to same CSV file

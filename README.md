# fast-bowling-ananlysis
analysis of fast bowling in round view angle
# Cricket Bowling Analysis System

AI-powered multi-camera fast bowling biomechanics analyser.
Records from 4 fixed cameras, detects pose with MediaPipe,
tracks the ball with YOLOv8 + DeepSORT, and computes
joint angles, velocities, phases, and release metrics.

---

## Requirements

- Python **3.9 or 3.10** (MediaPipe does not support 3.12 yet)
- Windows 10/11, macOS, or Linux
- 4 camera videos of the bowling action (front, back, left, right)

---

## Setup

### 1. Create virtual environment

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Install one at a time if any fail:
```
pip install opencv-python
pip install mediapipe
pip install numpy
pip install scipy
pip install pyyaml
pip install ultralytics
pip install deep-sort-realtime
pip install pandas
pip install gradio
```

### 3. Add Windows crash fix

Already included at the top of `main.py` and `app.py`:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

---

## Folder Structure

```
cricket_bowling_analysis/
├── config/config.yaml          ← all settings live here
├── data/
│   ├── raw/sessions/           ← drop 4 .mp4 files here
│   └── processed/sessions/     ← results saved here
├── outputs/                    ← annotated videos saved here
├── src/
│   ├── ingestion/              ← load session, extract frames
│   ├── sync/                   ← LED flash sync
│   ├── pose/                   ← MediaPipe pose detection
│   ├── ball/                   ← YOLOv8 + DeepSORT ball tracking
│   ├── biomechanics/           ← angles, velocity, phases, release
│   ├── storage/                ← CSV (now), Parquet/PostgreSQL (future)
│   ├── visualization/          ← skeleton, ball trail, HUD overlays
│   └── utils/                  ← config loader, video utils, math utils
├── app.py                      ← Gradio web UI
├── main.py                     ← CLI entry point
└── requirements.txt
```

---

## Usage

### Option A — Web UI (easiest)

```
python app.py
```
Open `http://localhost:7860` in browser.
Upload 4 videos → enter session name → click Analyse.

### Option B — Command line

```
python main.py --session data/raw/sessions/session_001
python main.py --session data/raw/sessions/session_001 --camera front
python main.py --session data/raw/sessions/session_001 --no-ball
```

---

## Camera Setup

```
        [FRONT]
           |
[LEFT] -- bowler -- [RIGHT]
           |
        [BACK]
```

- Mount all 4 cameras at stumps height (~0.7m)
- Fire a bright LED flash visible to all cameras at session start
- Record at 30fps, same resolution on all cameras

---

## Outputs

- **Annotated video**: `outputs/{session_id}_{camera}_analysis.avi`
  - Skeleton overlay (colour-coded joints)
  - Phase badge (RUN-UP / LOAD-UP / DELIVERY / FOLLOW-THROUGH)
  - Release frame flash
  - Live metrics HUD (angles + velocities)
  - Ball trail

- **Biomechanics CSV**: `data/processed/sessions/{session_id}/results/deliveries.csv`
  - One row per delivery
  - All joint angles at release
  - Peak angles across full delivery
  - Arm velocity, run-up speed, ball speed
  - Phase frame ranges

---

## Extending the System

| Goal | Where to add |
|------|-------------|
| New joint angle | `src/biomechanics/angle_calculator.py` |
| New velocity metric | `src/biomechanics/velocity_estimator.py` |
| New KPI in output | `src/biomechanics/feature_aggregator.py` |
| New HUD metric | `src/visualization/metrics_overlay.py` |
| Switch to Parquet | Change `storage.backend` in `config.yaml` |
| Switch to PostgreSQL | Change `storage.backend` to `db` in `config.yaml` |

---

## Common Windows Issues

| Problem | Fix |
|---------|-----|
| `mediapipe not found` | Activate venv first: `venv\Scripts\activate` |
| `python not recognized` | Reinstall Python, tick "Add to PATH" |
| `DLL load failed` | Install Visual C++ Redistributable from microsoft.com |
| `deep-sort-realtime fails` | `pip install deep-sort-realtime --no-deps` |
| Video won't play | Output is `.avi` (XVID codec) — open in VLC |

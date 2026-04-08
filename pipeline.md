┌─────────────────────────────────────────────────────────┐
│                    INPUT                                 │
│   front.mp4  back.mp4  left.mp4  right.mp4              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — session_loader.py                             │
│  Validates 4 files exist, reads fps/resolution          │
│  → SessionConfig dict                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — flash_sync.py + frame_aligner.py              │
│  Detects LED flash frame in each camera                 │
│  → sync_offsets {"front":12, "back":15, ...}            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3 — frame_extractor.py                            │
│  Trims all 4 cameras to common start point              │
│  → all_frames {"front": [...], "back": [...], ...}      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────┐       ┌──────────────────────┐
│  POSE BRANCH     │       │  BALL BRANCH         │
│                  │       │                      │
│ mediapipe_       │       │ yolo_detector.py     │
│ detector.py      │       │ → bbox per frame     │
│ → 33 landmarks   │       │                      │
│   per frame      │       │ deepsort_tracker.py  │
│                  │       │ → (cx,cy) per frame  │
│ keypoint_        │       │                      │
│ smoother.py      │       │ trajectory_fitter.py │
│ → smoothed       │       │ → ball_speed         │
│   landmarks      │       │   release_angle      │
└────────┬─────────┘       └──────────┬───────────┘
         │                            │
         └─────────────┬──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — BIOMECHANICS                                  │
│                                                         │
│  velocity_estimator.py                                  │
│  → arm_velocity, hip_velocity per frame                 │
│                                                         │
│  phase_segmenter.py                                     │
│  → every frame labelled:                               │
│     RUN-UP / LOAD-UP / DELIVERY / FOLLOW-THROUGH        │
│                                                         │
│  release_detector.py                                    │
│  → release_frame index                                  │
│                                                         │
│  angle_calculator.py                                    │
│  → 7 angles per frame + angles at release               │
│                                                         │
│  feature_aggregator.py                                  │
│  → DeliveryRecord dict (all KPIs combined)              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5 — STORAGE                                       │
│                                                         │
│  csv_writer.py        ← ACTIVE NOW                      │
│  parquet_writer.py    ← future (ML training)            │
│  db/delivery_repo.py  ← future (scale)                  │
│                                                         │
│  → deliveries.csv (one row per delivery)                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6 — VISUALIZATION                                 │
│                                                         │
│  skeleton_drawer.py   → colour-coded joints + bones     │
│  phase_annotator.py   → phase badge + release flash     │
│  metrics_overlay.py   → live HUD (angles, velocity)     │
│  ball_trail_drawer.py → ball + 20-frame fading trail    │
│                                                         │
│  → {session_id}_{camera}_analysis.avi                   │
└─────────────────────────────────────────────────────────┘
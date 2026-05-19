# LSTM Repeatability Model Notes

## Training Data

Each complete bowling delivery becomes one sequence:

```text
120 timesteps x 16 features
```

One delivery is one LSTM sample. The model is not trained on random frame rows.

## Why 120 Timesteps

The 7 side-on phases are resampled to fixed lengths:

```text
approach: 20
jump_bound: 15
bfc_window: 10
bfc_to_ffc: 20
ffc_window: 10
ffc_to_release: 25
follow_through: 20
```

Total:

```text
20 + 15 + 10 + 20 + 10 + 25 + 20 = 120
```

## Features

The model uses `LSTM_FEATURES` from `config.py`:

```text
front_knee_angle
back_knee_angle
trunk_lean_angle
bowling_arm_angle
bowling_elbow_angle
hip_center_x
hip_center_y
head_x
head_y
wrist_speed
hip_speed
front_knee_angle_velocity
arm_angle_velocity
trunk_lean_velocity
phase_id
normalized_phase_time
```

If a feature is missing from the source CSV, it is filled with `0` and a warning is printed.

## What The LSTM Learns

The LSTM learns the motion pattern through:

```text
approach
jump/bound
BFC
BFC to FFC
FFC
FFC to release
follow-through
```

## Output

The model predicts:

```text
final repeatability/stability potential
7 phase scores
```

Scores are sigmoid outputs shown as `0-100`.

## Important Wording

From one delivery, the model predicts repeatability potential, not proven actual repeatability.

Correct:

```text
High repeatability potential
```

Incorrect:

```text
This bowler is definitely repeatable
```

## Train/Test Split

Training and testing should use the same preprocessing pipeline.

The test CSV should come from the same format/place as the training CSVs.

Train/test split should be by bowler to avoid leakage.

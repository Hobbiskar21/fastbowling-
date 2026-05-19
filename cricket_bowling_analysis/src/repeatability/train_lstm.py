"""Train the side-on repeatability LSTM."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .lstm_model import BowlingRepeatabilityLSTM


def train_lstm_model(dataset_dir, output_model_path, epochs: int = 30, batch_size: int = 16, lr: float = 0.001):
    """Train binary overall repeatability classifier and save checkpoint."""
    dataset = Path(dataset_dir)
    x_train = np.load(dataset / "X_train.npy").astype(np.float32)
    y_train = np.load(dataset / "y_train.npy").astype(np.float32).reshape(-1, 1)
    x_test = np.load(dataset / "X_test.npy").astype(np.float32) if (dataset / "X_test.npy").exists() else np.empty((0,))
    y_test = np.load(dataset / "y_test.npy").astype(np.float32).reshape(-1, 1) if (dataset / "y_test.npy").exists() else np.empty((0, 1))

    if len(x_train) == 0:
        raise ValueError("No training samples found in dataset.")

    model = BowlingRepeatabilityLSTM(input_size=x_train.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if len(x_test):
            model.eval()
            with torch.no_grad():
                pred, _ = model(torch.tensor(x_test))
                row["test_loss"] = float(criterion(pred, torch.tensor(y_test)).item())
        print(f"[LSTM] Epoch {epoch}/{epochs}: {row}")
        history.append(row)

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": int(x_train.shape[-1]),
    }, out_path)
    history_path = out_path.parent / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"[LSTM] Saved model: {out_path}")
    print(f"[LSTM] Saved history: {history_path}")
    return model

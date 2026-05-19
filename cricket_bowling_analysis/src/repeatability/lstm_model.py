"""PyTorch LSTM model for side-on repeatability potential."""

import torch
from torch import nn


class BowlingRepeatabilityLSTM(nn.Module):
    """Predict one overall score and 7 phase scores from a delivery sequence."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.overall_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.phase_head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 7), nn.Sigmoid())

    def forward(self, x):
        """Return (overall_score, phase_scores), each in 0..1."""
        _, (hidden, _) = self.lstm(x)
        last = hidden[-1]
        return self.overall_head(last), self.phase_head(last)

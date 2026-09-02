"""
Opacus-Compatible Sequence and Convolutional Models for Clinical Deterioration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from opacus.layers import DPGRU, DPLSTM


class DPLSTMClassifier(nn.Module):
    """
    Differentially private LSTM sequence classifier.
    Consumes 12 timesteps × 6 vitals and outputs raw scalar logits.
    """

    def __init__(
        self,
        inp_dim: int = 6,
        hid_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = DPLSTM(
            inp_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 12, 6)
        out, _ = self.rnn(x)
        # Take last timestep's hidden state
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class DPGRUClassifier(nn.Module):
    """
    Differentially private GRU sequence classifier.
    Consumes 12 timesteps × 6 vitals and outputs raw scalar logits.
    """

    def __init__(
        self,
        inp_dim: int = 6,
        hid_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = DPGRU(
            inp_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class CNN1DClassifier(nn.Module):
    """
    Temporal 1D Convolutional classifier with Opacus-compatible GroupNorm.
    Consumes 12 timesteps × 6 vitals and outputs raw scalar logits.
    """

    def __init__(
        self,
        inp_dim: int = 6,
        conv_dims: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_c = inp_dim

        for out_c in conv_dims:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2))
            # GroupNorm is natively supported by Opacus per-sample gradient computation
            layers.append(nn.GroupNorm(num_groups=min(4, out_c), num_channels=out_c))
            layers.append(nn.ReLU())
            in_c = out_c

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_dims[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 12, 6) -> transpose to (batch_size, 6, 12) for Conv1d
        x_trans = x.transpose(1, 2)
        feat = self.features(x_trans).squeeze(-1)
        logits = self.head(feat).squeeze(-1)
        return logits


def get_model(
    architecture: str = "DPLSTM",
    inp_dim: int = 6,
    hid_dim: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory helper to instantiate detector architectures by name."""
    arch_lower = architecture.lower()
    if "lstm" in arch_lower:
        return DPLSTMClassifier(inp_dim=inp_dim, hid_dim=hid_dim, dropout=dropout)
    elif "gru" in arch_lower:
        return DPGRUClassifier(inp_dim=inp_dim, hid_dim=hid_dim, dropout=dropout)
    elif "cnn" in arch_lower:
        return CNN1DClassifier(inp_dim=inp_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from 'DPLSTM', 'DPGRU', 'CNN1D'.")

from __future__ import annotations

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """A simple expert head that maps shared features to Q-values."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

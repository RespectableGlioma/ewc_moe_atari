from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RouterGRU(nn.Module):
    """A small GRU router that maintains a hidden state across timesteps.

    Input: feature vector f_t from an encoder.
    Output: gating logits over experts.

    This is intentionally lightweight; the goal is that the hidden state captures
    latent context (which Atari game, subtask phase, etc.).
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)

        self.gru = nn.GRUCell(self.feature_dim, self.hidden_dim)
        self.to_logits = nn.Linear(self.hidden_dim, self.num_experts)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step router update.

        Args:
            features: (B, feature_dim)
            hidden: (B, hidden_dim)

        Returns:
            logits: (B, num_experts)
            new_hidden: (B, hidden_dim)
        """
        new_hidden = self.gru(features, hidden)
        logits = self.to_logits(new_hidden)
        return logits, new_hidden

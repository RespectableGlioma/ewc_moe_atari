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

    def __init__(self, feature_dim: int, hidden_dim: int, num_experts: int, noise_std: float = 1.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.noise_std = float(noise_std)

        self.gru = nn.GRUCell(self.feature_dim, self.hidden_dim)
        self.to_logits = nn.Linear(self.hidden_dim, self.num_experts)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step router update.

        Args:
            features: (B, feature_dim)
            hidden: (B, hidden_dim)
            training: whether to inject noise

        Returns:
            logits: (B, num_experts)
            new_hidden: (B, hidden_dim)
        """
        new_hidden = self.gru(features, hidden)
        logits = self.to_logits(new_hidden)

        if training and self.noise_std > 0.0:
            # Inject noise to encourage exploration
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        return logits, new_hidden

    def compute_load_balancing_loss(self, gating_probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Computes the auxiliary load balancing loss (Switch Transformer style).

        Args:
            gating_probs: (B, num_experts)
            topk_idx: (B, k)

        Returns:
            scalar loss tensor
        """
        B, E = gating_probs.shape
        k = topk_idx.shape[1]

        # Calculate fraction of samples dispatched to each expert
        # We treat each of the k selections as a dispatch event.
        flat_idx = topk_idx.flatten()
        
        with torch.no_grad():
            counts = torch.zeros(E, device=gating_probs.device)
            ones = torch.ones(flat_idx.size(0), device=gating_probs.device)
            counts.index_add_(0, flat_idx, ones)
            # fraction dispatched: shape (E,)
            fraction = counts / float(flat_idx.numel())

        # Average gating probability per expert across the batch
        mean_probs = gating_probs.mean(dim=0)  # (E,)

        # Loss = N * dot(fraction, mean_probs)
        # This is minimized when both distributions are uniform.
        loss = self.num_experts * (fraction * mean_probs).sum()
        return loss

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ExpertMLP
from .router_rnn import RouterGRU


class ConvEncoder(nn.Module):
    """Classic DQN-style convolutional encoder for 84x84 inputs.

    Assumes input is in (B, C, H, W) with dtype uint8 or float.
    """

    def __init__(self, in_channels: int, feature_dim: int = 512):
        super().__init__()
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 84x84 -> 20x20 -> 9x9 -> 7x7 with these convs
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Normalize if input is bytes
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        x = self.conv(obs)
        x = self.fc(x)
        return x


@dataclass
class MoEOutput:
    q_values: torch.Tensor              # (B, action_dim)
    gating_probs: torch.Tensor          # (B, num_experts)
    hidden: torch.Tensor                # (B, hidden_dim)


class MoEDQN(nn.Module):
    """Mixture-of-Experts DQN where a recurrent router gates multiple Q-head experts."""

    def __init__(
        self,
        *,
        obs_stack: int,
        action_dim: int,
        num_experts: int = 8,
        router_hidden_dim: int = 128,
        expert_hidden_dim: int = 256,
        feature_dim: int = 512,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.obs_stack = int(obs_stack)
        self.action_dim = int(action_dim)
        self.num_experts = int(num_experts)
        self.router_hidden_dim = int(router_hidden_dim)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.feature_dim = int(feature_dim)
        self.temperature = float(temperature)

        self.encoder = ConvEncoder(in_channels=self.obs_stack, feature_dim=self.feature_dim)
        self.router = RouterGRU(feature_dim=self.feature_dim, hidden_dim=self.router_hidden_dim, num_experts=self.num_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(self.feature_dim, self.action_dim, hidden_dim=self.expert_hidden_dim)
            for _ in range(self.num_experts)
        ])

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.router.init_hidden(batch_size, device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> MoEOutput:
        """One-step forward.

        Args:
            obs: (B, stack, 84, 84)
            hidden: (B, router_hidden_dim)
        """
        feats = self.encoder(obs)
        logits, new_hidden = self.router(feats, hidden)
        gating_probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)

        # Compute all experts (okay for small num_experts). Later, replace with top-k for scale.
        q_all = torch.stack([expert(feats) for expert in self.experts], dim=1)  # (B, E, A)
        q_values = (gating_probs.unsqueeze(-1) * q_all).sum(dim=1)

        return MoEOutput(q_values=q_values, gating_probs=gating_probs, hidden=new_hidden)

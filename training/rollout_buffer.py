"""
RolloutBuffer: Storage for PPO rollouts.

Stores observations, actions, rewards, values, and log probabilities
for PPO training updates.
"""

import torch
import numpy as np
from typing import Generator, Tuple, Optional


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.

    Supports GAE-lambda advantage computation.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        device: torch.device = torch.device("cuda"),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs

        self.pos = 0
        self.full = False

        # Allocate buffers
        self.observations = torch.zeros(
            (buffer_size, n_envs) + obs_shape,
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.long,
            device=device,
        )
        self.rewards = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.dones = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.values = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.log_probs = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.advantages = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.returns = torch.zeros(
            (buffer_size, n_envs),
            dtype=torch.float32,
            device=device,
        )

    def reset(self):
        """Reset buffer position."""
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """
        Add a step to the buffer.

        All inputs should be tensors with shape (n_envs,) or (n_envs, ...).
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
    ):
        """
        Compute returns and advantages using GAE-lambda.

        Args:
            last_values: (n_envs,) value estimates for last state
            last_dones: (n_envs,) done flags for last state
        """
        last_gae_lam = 0

        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - last_dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

            self.advantages[step] = last_gae_lam

        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

    def get_samples(
        self,
        batch_size: int,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random minibatches for training.

        Yields:
            (observations, actions, old_values, old_log_probs, advantages, returns)
        """
        size = self.pos * self.n_envs
        indices = torch.randperm(size, device=self.device)

        # Flatten buffers
        obs_flat = self.observations[:self.pos].reshape(-1, *self.obs_shape)
        actions_flat = self.actions[:self.pos].reshape(-1)
        values_flat = self.values[:self.pos].reshape(-1)
        log_probs_flat = self.log_probs[:self.pos].reshape(-1)
        advantages_flat = self.advantages[:self.pos].reshape(-1)
        returns_flat = self.returns[:self.pos].reshape(-1)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (
            advantages_flat.std() + 1e-8
        )

        for start in range(0, size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                obs_flat[batch_indices],
                actions_flat[batch_indices],
                values_flat[batch_indices],
                log_probs_flat[batch_indices],
                advantages_flat[batch_indices],
                returns_flat[batch_indices],
            )

    def __len__(self):
        return self.pos * self.n_envs

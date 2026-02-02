"""
PPOTrainer: Proximal Policy Optimization for expert training.

Standard PPO implementation for training expert networks within
the day phase of the bi-level training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Optional, Tuple
import logging

from .rollout_buffer import RolloutBuffer
from core.expert import Expert

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO trainer for expert networks.

    Implements the inner loop (day phase) of the hierarchical training.
    """

    def __init__(
        self,
        expert: Expert,
        device: torch.device = torch.device("cuda"),
        learning_rate: float = 2.5e-4,
        n_steps: int = 128,
        batch_size: int = 256,
        n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.1,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.01,
    ):
        self.expert = expert
        self.device = device

        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Optimizer
        self.optimizer = Adam(expert.parameters(), lr=learning_rate, eps=1e-5)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            obs_shape=expert.obs_shape,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Statistics
        self.total_timesteps = 0
        self.num_updates = 0

    def set_expert(self, expert: Expert):
        """Switch to a different expert."""
        self.expert = expert
        self.optimizer = Adam(expert.parameters(), lr=self.learning_rate, eps=1e-5)
        self.buffer.reset()

    def collect_rollout(
        self,
        env,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Collect rollout data from environment.

        Args:
            env: Gymnasium environment
            obs: (1, C, H, W) initial observation

        Returns:
            last_obs: final observation
            info: dict with episode statistics
        """
        self.buffer.reset()
        self.expert.eval()

        episode_rewards = []
        episode_lengths = []
        current_reward = 0
        current_length = 0

        for _ in range(self.n_steps):
            with torch.no_grad():
                action, log_prob, value = self.expert.get_action(obs)

            # Step environment
            action_np = action.cpu().numpy()[0]
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            current_reward += reward
            current_length += 1

            # Convert to tensors
            reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

            # Store in buffer
            self.buffer.add(
                obs=obs.squeeze(0),
                action=action,
                reward=reward_t,
                done=done_t,
                value=value,
                log_prob=log_prob,
            )

            self.total_timesteps += 1

            # Prepare next observation
            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                current_reward = 0
                current_length = 0
                next_obs_np, _ = env.reset()

            obs = torch.tensor(
                next_obs_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

        # Compute last value for GAE
        with torch.no_grad():
            last_value = self.expert.get_value(obs)
            last_done = torch.tensor(
                [float(done)],
                dtype=torch.float32,
                device=self.device,
            )

        self.buffer.compute_returns_and_advantages(last_value, last_done)

        info = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
        }

        return obs, info

    def train(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.

        Returns:
            Dictionary of training metrics
        """
        self.expert.train()

        clip_fractions = []
        value_losses = []
        policy_losses = []
        entropy_losses = []
        approx_kls = []

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_samples(self.batch_size):
                (
                    obs,
                    actions,
                    old_values,
                    old_log_probs,
                    advantages,
                    returns,
                ) = batch

                # Evaluate current policy
                log_probs, values, entropy = self.expert.evaluate_actions(obs, actions)

                # Ratio for PPO clipping
                ratio = torch.exp(log_probs - old_log_probs)

                # Policy loss
                policy_loss1 = advantages * ratio
                policy_loss2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss1 = F.mse_loss(values, returns, reduction='none')
                    value_loss2 = F.mse_loss(values_clipped, returns, reduction='none')
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = F.mse_loss(values, returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.expert.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Statistics
                clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()
                clip_fractions.append(clip_fraction.item())
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())

                # KL divergence approximation
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

            # Early stopping based on KL
            if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                logger.debug(f"Early stopping at epoch {epoch} due to KL divergence")
                break

        self.num_updates += 1

        return {
            'policy_loss': sum(policy_losses) / len(policy_losses),
            'value_loss': sum(value_losses) / len(value_losses),
            'entropy_loss': sum(entropy_losses) / len(entropy_losses),
            'clip_fraction': sum(clip_fractions) / len(clip_fractions),
            'approx_kl': sum(approx_kls) / len(approx_kls),
            'total_timesteps': self.total_timesteps,
        }

    def step(
        self,
        env,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one rollout + training step.

        Args:
            env: Gymnasium environment
            obs: Initial observation

        Returns:
            last_obs: Final observation
            metrics: Combined rollout and training metrics
        """
        # Collect rollout
        last_obs, rollout_info = self.collect_rollout(env, obs)

        # Train on rollout
        train_metrics = self.train()

        metrics = {
            **rollout_info,
            **train_metrics,
        }

        return last_obs, metrics

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action from current expert."""
        self.expert.eval()
        with torch.no_grad():
            action, _, _ = self.expert.get_action(obs, deterministic)
        return action

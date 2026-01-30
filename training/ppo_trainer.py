"""
Training infrastructure for Out-of-Core MoE RL.

Implements:
1. PPO algorithm adapted for MoE
2. Sequential game training with environment switching
3. Expert specialization tracking
4. Cache performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
import time

from ..models.moe_agent import MoERLAgent, MoERLAgentConfig
from ..core.env_detector import EnvironmentDetectorTrainer


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[Tensor]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    expert_ids: List[List[int]]
    env_ids: List[int]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.expert_ids = []
        self.env_ids = []
    
    def add(
        self,
        obs: Tensor,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        expert_ids: List[int],
        env_id: int,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.expert_ids.append(expert_ids)
        self.env_ids.append(env_id)
    
    def __len__(self):
        return len(self.observations)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE returns and advantages."""
        returns = []
        advantages = []
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = (
                self.rewards[t] + 
                gamma * next_value * next_non_terminal - 
                self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return returns, advantages


class PPOTrainer:
    """
    PPO trainer adapted for Out-of-Core MoE.
    
    Key adaptations:
    1. Tracks expert usage per environment
    2. Trains environment detector alongside policy
    3. Monitors cache performance
    4. Supports sequential game training
    """
    
    def __init__(
        self,
        agent: MoERLAgent,
        config: MoERLAgentConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        aux_loss_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.agent = agent
        self.config = config
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.aux_loss_coef = aux_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        
        # Optimizer (excludes expert params - they're handled separately)
        self.optimizer = torch.optim.Adam(
            agent.parameters(),  # This gets shared params
            lr=lr,
        )
        
        # Environment detector trainer
        self.detector_trainer = EnvironmentDetectorTrainer(
            agent.env_detector,
            lr=lr * 0.1,  # Lower LR for detector
        )
        
        # Tracking
        self.expert_usage_by_env: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.episode_rewards: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.cache_hit_history = deque(maxlen=1000)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()
    
    def collect_rollout(
        self,
        env,
        env_id: int,
        num_steps: int,
        context_len: int,
    ) -> Dict[str, float]:
        """
        Collect rollout data from a single environment.
        
        Args:
            env: Gymnasium environment
            env_id: ID of current environment
            num_steps: Steps to collect
            context_len: Context window length
            
        Returns:
            Dictionary of rollout statistics
        """
        self.agent.eval()
        self.rollout_buffer.clear()
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
        
        # Context window for transformer
        obs_buffer = deque(maxlen=context_len)
        obs_tensor = self._preprocess_obs(obs)
        for _ in range(context_len):
            obs_buffer.append(obs_tensor)
        
        total_reward = 0
        episode_reward = 0
        num_episodes = 0
        
        device = next(self.agent.parameters()).device
        
        for step in range(num_steps):
            # Build context tensor
            context = torch.stack(list(obs_buffer), dim=1).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                output = self.agent.forward(context, env_id=env_id, prefetch=True)
                
                # Sample action
                probs = F.softmax(output.action_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Track expert usage
            for expert_id in output.expert_ids:
                self.expert_usage_by_env[env_id][expert_id] += 1
            
            # Track cache performance
            cache_stats = output.cache_stats
            self.cache_hit_history.append(cache_stats['hit_rate'])
            
            # Store for detector training
            self.detector_trainer.store_sample(
                context[:, -1].squeeze(0).cpu(),
                env_id,
                output.expert_ids
            )
            
            # Environment step
            next_obs, reward, done, truncated, info = env.step(action.item())
            done = done or truncated
            
            # Store in buffer
            self.rollout_buffer.add(
                obs=context.cpu(),
                action=action.item(),
                reward=reward,
                value=output.value.item(),
                log_prob=log_prob.item(),
                done=done,
                expert_ids=output.expert_ids,
                env_id=env_id,
            )
            
            episode_reward += reward
            total_reward += reward
            
            # Update observation
            obs = next_obs
            obs_tensor = self._preprocess_obs(obs)
            obs_buffer.append(obs_tensor)
            
            if done:
                self.episode_rewards[env_id].append(episode_reward)
                episode_reward = 0
                num_episodes += 1
                
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs_tensor = self._preprocess_obs(obs)
                obs_buffer.clear()
                for _ in range(context_len):
                    obs_buffer.append(obs_tensor)
        
        # Get last value for GAE
        with torch.no_grad():
            context = torch.stack(list(obs_buffer), dim=1).unsqueeze(0).to(device)
            output = self.agent.forward(context, env_id=env_id, prefetch=False)
            last_value = output.value.item()
        
        # Compute returns and advantages
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        return {
            'total_reward': total_reward,
            'mean_episode_reward': np.mean(self.episode_rewards[env_id]) if self.episode_rewards[env_id] else 0,
            'num_episodes': num_episodes,
            'cache_hit_rate': np.mean(self.cache_hit_history) if self.cache_hit_history else 0,
            'returns': returns,
            'advantages': advantages,
        }
    
    def _preprocess_obs(self, obs: np.ndarray) -> Tensor:
        """Preprocess observation to tensor."""
        if obs.ndim == 2:
            obs = obs[np.newaxis, ...]  # Add channel dim
        return torch.from_numpy(obs.astype(np.float32) / 255.0)
    
    def train_on_rollout(
        self,
        returns: List[float],
        advantages: List[float],
    ) -> Dict[str, float]:
        """
        Train on collected rollout data using PPO.
        
        Returns:
            Dictionary of training losses
        """
        self.agent.train()
        
        device = next(self.agent.parameters()).device
        
        # Convert to tensors
        obs = torch.cat([o for o in self.rollout_buffer.observations], dim=0).to(device)
        actions = torch.tensor(self.rollout_buffer.actions, device=device)
        old_log_probs = torch.tensor(self.rollout_buffer.log_probs, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_aux_loss = 0
        num_updates = 0
        
        batch_size = len(self.rollout_buffer)
        
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                
                # Forward pass
                output = self.agent.forward(mb_obs, prefetch=False)
                
                # Policy loss (PPO clipped objective)
                probs = F.softmax(output.action_logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(output.value.squeeze(-1), mb_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy +
                    self.aux_loss_coef * output.aux_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_aux_loss += output.aux_loss.item() if isinstance(output.aux_loss, Tensor) else output.aux_loss
                num_updates += 1
        
        # Train environment detector
        detector_losses = self.detector_trainer.train_step(batch_size=64)
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'aux_loss': total_aux_loss / num_updates,
            'detector_loss': detector_losses['total_loss'],
        }
    
    def get_expert_specialization_report(self) -> Dict:
        """Generate report on expert specialization by environment."""
        report = {}
        
        for env_id, expert_counts in self.expert_usage_by_env.items():
            total_usage = sum(expert_counts.values())
            top_experts = sorted(
                expert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            report[env_id] = {
                'total_expert_activations': total_usage,
                'unique_experts_used': len(expert_counts),
                'top_10_experts': [(e, c, c/total_usage) for e, c in top_experts],
                'concentration': self._compute_expert_concentration(expert_counts),
            }
        
        return report
    
    def _compute_expert_concentration(self, counts: Dict[int, int]) -> float:
        """Compute concentration of expert usage (higher = more specialized)."""
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(counts))
        
        # Return 1 - normalized entropy (so higher = more concentrated)
        return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


class SequentialGameTrainer:
    """
    Trainer for sequential multi-game training.
    
    This implements the core experimental setup:
    - Train on games sequentially (one at a time)
    - Track catastrophic forgetting (or lack thereof)
    - Monitor expert specialization emergence
    """
    
    def __init__(
        self,
        agent: MoERLAgent,
        config: MoERLAgentConfig,
        envs: List,  # List of gym environments
        env_names: List[str],
        steps_per_game: int = 100000,
        eval_interval: int = 10000,
        checkpoint_interval: int = 50000,
        log_dir: str = "./logs",
    ):
        self.agent = agent
        self.config = config
        self.envs = envs
        self.env_names = env_names
        self.steps_per_game = steps_per_game
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.log_dir = log_dir
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            agent=agent,
            config=config,
        )
        
        # Tracking
        self.all_rewards: Dict[int, List[float]] = defaultdict(list)
        self.forgetting_metrics: Dict[str, List[float]] = defaultdict(list)
        self.training_log = []
    
    def train(self, num_rounds: int = 1):
        """
        Train on all games sequentially.
        
        Args:
            num_rounds: Number of times to cycle through all games
        """
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1}/{num_rounds} ===")
            
            for env_idx, (env, env_name) in enumerate(zip(self.envs, self.env_names)):
                print(f"\n--- Training on {env_name} (Game {env_idx + 1}/{len(self.envs)}) ---")
                
                self._train_on_game(
                    env=env,
                    env_id=env_idx,
                    env_name=env_name,
                    round_idx=round_idx,
                )
                
                # Evaluate on all previous games to measure forgetting
                if env_idx > 0:
                    self._evaluate_forgetting(env_idx, round_idx)
    
    def _train_on_game(
        self,
        env,
        env_id: int,
        env_name: str,
        round_idx: int,
    ):
        """Train on a single game."""
        steps_done = 0
        rollout_steps = 2048
        
        while steps_done < self.steps_per_game:
            # Collect rollout
            rollout_stats = self.ppo_trainer.collect_rollout(
                env=env,
                env_id=env_id,
                num_steps=rollout_steps,
                context_len=self.config.context_len,
            )
            
            # Train
            train_stats = self.ppo_trainer.train_on_rollout(
                rollout_stats['returns'],
                rollout_stats['advantages'],
            )
            
            steps_done += rollout_steps
            
            # Log
            log_entry = {
                'round': round_idx,
                'env_id': env_id,
                'env_name': env_name,
                'steps': steps_done,
                'reward': rollout_stats['mean_episode_reward'],
                'cache_hit_rate': rollout_stats['cache_hit_rate'],
                **train_stats,
            }
            self.training_log.append(log_entry)
            
            # Track rewards
            self.all_rewards[env_id].append(rollout_stats['mean_episode_reward'])
            
            # Print progress
            if steps_done % self.eval_interval == 0:
                print(f"  Steps: {steps_done}/{self.steps_per_game} | "
                      f"Reward: {rollout_stats['mean_episode_reward']:.1f} | "
                      f"Cache Hit: {rollout_stats['cache_hit_rate']:.2%}")
    
    def _evaluate_forgetting(self, current_env_idx: int, round_idx: int):
        """Evaluate performance on previously trained games."""
        print("\n  Evaluating retention on previous games...")
        
        for prev_idx in range(current_env_idx):
            env = self.envs[prev_idx]
            env_name = self.env_names[prev_idx]
            
            # Quick evaluation
            eval_reward = self._quick_eval(env, prev_idx, num_episodes=5)
            
            # Compare to best performance
            if self.all_rewards[prev_idx]:
                best_reward = max(self.all_rewards[prev_idx])
                retention = eval_reward / (best_reward + 1e-8)
            else:
                retention = 1.0
            
            self.forgetting_metrics[env_name].append(retention)
            print(f"    {env_name}: {eval_reward:.1f} (retention: {retention:.2%})")
    
    def _quick_eval(self, env, env_id: int, num_episodes: int = 5) -> float:
        """Quick evaluation on an environment."""
        self.agent.eval()
        device = next(self.agent.parameters()).device
        
        total_reward = 0
        obs_buffer = deque(maxlen=self.config.context_len)
        
        for _ in range(num_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            obs_tensor = self.ppo_trainer._preprocess_obs(obs)
            obs_buffer.clear()
            for _ in range(self.config.context_len):
                obs_buffer.append(obs_tensor)
            
            episode_reward = 0
            done = False
            
            while not done:
                context = torch.stack(list(obs_buffer), dim=1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = self.agent.forward(context, env_id=env_id, prefetch=False)
                    action = output.action_logits.argmax(dim=-1).item()
                
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                episode_reward += reward
                
                obs_tensor = self.ppo_trainer._preprocess_obs(obs)
                obs_buffer.append(obs_tensor)
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def get_summary_report(self) -> Dict:
        """Generate summary report of training."""
        expert_report = self.ppo_trainer.get_expert_specialization_report()
        
        # Compute cross-game expert overlap
        env_experts = {}
        for env_id, expert_counts in self.ppo_trainer.expert_usage_by_env.items():
            top_experts = set(
                e for e, c in sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            env_experts[env_id] = top_experts
        
        # Pairwise overlap
        overlap_matrix = {}
        env_ids = list(env_experts.keys())
        for i, env_i in enumerate(env_ids):
            for j, env_j in enumerate(env_ids):
                if i < j:
                    overlap = len(env_experts[env_i] & env_experts[env_j])
                    overlap_matrix[(env_i, env_j)] = overlap
        
        return {
            'expert_specialization': expert_report,
            'expert_overlap': overlap_matrix,
            'forgetting_metrics': dict(self.forgetting_metrics),
            'final_rewards': {env_id: rewards[-1] if rewards else 0 
                            for env_id, rewards in self.all_rewards.items()},
            'cache_stats': self.agent.expert_store.get_stats_summary(),
        }

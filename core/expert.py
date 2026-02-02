"""
Expert: Game-playing network that gets swapped in/out of GPU memory.

Each expert is a complete actor-critic network for playing a specific
game or set of similar games.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, Any
import numpy as np


class NatureCNN(nn.Module):
    """
    CNN backbone from DQN Nature paper.
    Standard architecture for Atari feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 4,
        feature_dim: int = 512,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 64 * 7 * 7 = 3136 for 84x84 input
        self.fc = nn.Sequential(
            nn.Linear(3136, feature_dim),
            nn.ReLU(),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.conv(x))


class Expert(nn.Module):
    """
    Actor-Critic expert for game-specific policy.

    Designed to be swapped in/out of GPU memory as needed.
    ~50M params -> ~200MB when stored.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        num_actions: int = 18,  # Max Atari actions
        feature_dim: int = 512,
        expert_id: Optional[str] = None,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.expert_id = expert_id or "unnamed"

        # Shared CNN backbone
        self.backbone = NatureCNN(
            in_channels=obs_shape[0],
            feature_dim=feature_dim,
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Metadata for tracking
        self.metadata: Dict[str, Any] = {
            'games_trained': [],
            'total_frames': 0,
            'total_episodes': 0,
            'best_episode_reward': float('-inf'),
            'creation_time': None,
        }

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # Smaller init for policy head
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy logits and value.

        Args:
            obs: (batch, C, H, W) observations

        Returns:
            policy_logits: (batch, num_actions)
            value: (batch, 1)
        """
        features = self.backbone(obs)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: (batch, C, H, W) observations
            deterministic: if True, take argmax instead of sampling

        Returns:
            action: (batch,) sampled actions
            log_prob: (batch,) log probabilities
            value: (batch,) value estimates
        """
        policy_logits, value = self.forward(obs)
        dist = Categorical(logits=policy_logits)

        if deterministic:
            action = policy_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Args:
            obs: (batch, C, H, W) observations
            actions: (batch,) actions taken

        Returns:
            log_prob: (batch,) log probabilities of actions
            value: (batch,) value estimates
            entropy: (batch,) policy entropy
        """
        policy_logits, value = self.forward(obs)
        dist = Categorical(logits=policy_logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value.squeeze(-1), entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.backbone(obs)
        return self.critic(features).squeeze(-1)

    def update_metadata(
        self,
        game_name: Optional[str] = None,
        frames: int = 0,
        episodes: int = 0,
        episode_reward: Optional[float] = None,
    ):
        """Update expert metadata after training."""
        if game_name and game_name not in self.metadata['games_trained']:
            self.metadata['games_trained'].append(game_name)

        self.metadata['total_frames'] += frames
        self.metadata['total_episodes'] += episodes

        if episode_reward is not None:
            self.metadata['best_episode_reward'] = max(
                self.metadata['best_episode_reward'],
                episode_reward
            )

    def get_state_dict_with_metadata(self) -> Dict[str, Any]:
        """Get state dict including metadata for saving."""
        return {
            'state_dict': self.state_dict(),
            'metadata': self.metadata,
            'expert_id': self.expert_id,
            'obs_shape': self.obs_shape,
            'num_actions': self.num_actions,
        }

    @classmethod
    def from_state_dict_with_metadata(
        cls,
        data: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
    ) -> 'Expert':
        """Load expert from saved state dict with metadata."""
        expert = cls(
            obs_shape=data['obs_shape'],
            num_actions=data['num_actions'],
            expert_id=data['expert_id'],
        )
        expert.load_state_dict(data['state_dict'])
        expert.metadata = data['metadata']
        expert.to(device)
        return expert

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ExpertEnsemble(nn.Module):
    """
    Ensemble of experts for potential parallel inference.
    Useful when prefetching multiple candidate experts.
    """

    def __init__(self, experts: list):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(
        self,
        obs: torch.Tensor,
        expert_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through specific expert."""
        return self.experts[expert_idx](obs)

    def get_action(
        self,
        obs: torch.Tensor,
        expert_idx: int,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from specific expert."""
        return self.experts[expert_idx].get_action(obs, deterministic)

"""
MetaRSSM: Abstract world model for game-of-games dynamics.

The meta-agent learns a higher-order MDP where states are "which game"
and transitions are "game switches." It compresses away game-specific
dynamics and retains only game-identity and transition structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, NamedTuple


class VectorQuantize(nn.Module):
    """
    VQ-VAE style discrete codebook for game prototypes.
    Maps continuous embeddings to discrete codes that serve as expert keys.
    """

    def __init__(
        self,
        codebook_size: int = 64,
        code_dim: int = 32,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        # Codebook: 64 "game prototypes"
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, code_dim) continuous embeddings

        Returns:
            quantized: (batch, code_dim) quantized embeddings
            indices: (batch,) codebook indices
            loss: commitment loss for training
        """
        # Compute distances to codebook
        # z: (B, D), codebook: (K, D)
        distances = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1)
        )

        # Find nearest codes
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)

        # Commitment loss
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        loss = self.commitment_cost * commitment_loss + codebook_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices, loss

    def get_code_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve embeddings for given indices."""
        return self.codebook(indices)


class CNNEncoder(nn.Module):
    """
    Small CNN encoder: frames -> abstract features.
    Not designed for pixel reconstruction - just game identity.
    """

    def __init__(
        self,
        in_channels: int = 4,  # frame stack
        out_dim: int = 256,
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

        # Compute flattened size for 84x84 input
        # After conv: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        # 64 * 7 * 7 = 3136
        self.fc = nn.Linear(3136, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, 84, 84) observations
        Returns:
            (batch, out_dim) encoded features
        """
        # Normalize to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.conv(x))


class MetaRSSMState(NamedTuple):
    """State container for MetaRSSM."""
    h: torch.Tensor          # GRU hidden state (batch, hidden_dim)
    z: torch.Tensor          # Current quantized code (batch, code_dim)
    code_idx: torch.Tensor   # Current code index (batch,)


class MetaRSSM(nn.Module):
    """
    Abstract world model for game-of-games dynamics.

    Trained unsupervised on observation stream during "day" phase.
    Reward signal from expert performance during "night" update.

    Key outputs:
    - h_t: GRU hidden state (temporal context)
    - z_t: VQ code (expert retrieval key)
    - KL(posterior || prior): Switch detection trigger (spike = game changed)
    - p(z_{t+1} | h_t): Prefetch distribution
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        hidden_dim: int = 256,
        code_dim: int = 32,
        codebook_size: int = 64,
        kl_threshold: float = 2.0,  # Threshold for switch detection
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.kl_threshold = kl_threshold

        # Encoder: frames -> abstract features
        self.encoder = CNNEncoder(
            in_channels=obs_shape[0],
            out_dim=hidden_dim,
        )

        # Recurrent backbone: integrates over time
        self.gru = nn.GRUCell(
            input_size=hidden_dim + code_dim,
            hidden_size=hidden_dim,
        )

        # VQ codebook for discrete game codes
        self.vq = VectorQuantize(
            codebook_size=codebook_size,
            code_dim=code_dim,
        )

        # Prior: p(z_t | h_t) - predict game from dynamics alone
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, codebook_size),
        )

        # Posterior: q(z_t | h_t, x_t) - infer game given observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, code_dim),  # Output continuous embedding for VQ
        )

        # Transition predictor: p(g_{t+1} | h_t, z_t) for prefetch
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim + code_dim, 128),
            nn.ReLU(),
            nn.Linear(128, codebook_size),
        )

        # Running baseline for night updates (REINFORCE)
        self.register_buffer("baseline", torch.tensor(0.0))
        self.register_buffer("baseline_count", torch.tensor(0))

    def init_state(self, batch_size: int, device: torch.device) -> MetaRSSMState:
        """Initialize hidden state."""
        return MetaRSSMState(
            h=torch.zeros(batch_size, self.hidden_dim, device=device),
            z=torch.zeros(batch_size, self.code_dim, device=device),
            code_idx=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def forward(
        self,
        obs: torch.Tensor,
        state: MetaRSSMState,
    ) -> Tuple[MetaRSSMState, dict]:
        """
        Process one timestep.

        Args:
            obs: (batch, C, H, W) observation
            state: previous MetaRSSMState

        Returns:
            new_state: updated MetaRSSMState
            outputs: dict with 'kl', 'prior_logits', 'transition_logits', 'vq_loss'
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Encode observation
        e_t = self.encoder(obs)

        # Prior from dynamics alone
        prior_logits = self.prior_net(state.h)
        prior_probs = F.softmax(prior_logits, dim=-1)

        # Posterior using observation
        post_input = torch.cat([state.h, e_t], dim=-1)
        post_embedding = self.posterior_net(post_input)

        # Quantize to get discrete code
        z_t, code_idx, vq_loss = self.vq(post_embedding)

        # Compute KL between posterior (one-hot from VQ) and prior
        # Since posterior is effectively one-hot after VQ, use cross-entropy
        post_one_hot = F.one_hot(code_idx, self.codebook_size).float()
        kl = F.kl_div(
            prior_probs.log(),
            post_one_hot,
            reduction='none'
        ).sum(dim=-1)

        # Update recurrent state
        gru_input = torch.cat([e_t, z_t], dim=-1)
        h_new = self.gru(gru_input, state.h)

        # Transition prediction for prefetch
        trans_input = torch.cat([h_new, z_t], dim=-1)
        transition_logits = self.transition_net(trans_input)

        new_state = MetaRSSMState(h=h_new, z=z_t, code_idx=code_idx)

        outputs = {
            'kl': kl,
            'prior_logits': prior_logits,
            'transition_logits': transition_logits,
            'vq_loss': vq_loss,
            'encoding': e_t,
        }

        return new_state, outputs

    def detect_switch(self, kl: torch.Tensor) -> torch.Tensor:
        """
        Detect game switch from KL divergence spike.

        Args:
            kl: (batch,) KL divergence values

        Returns:
            (batch,) boolean tensor indicating switch detected
        """
        return kl > self.kl_threshold

    def get_prefetch_distribution(
        self,
        state: MetaRSSMState,
    ) -> torch.Tensor:
        """
        Get distribution over next game codes for prefetching.

        Args:
            state: current MetaRSSMState

        Returns:
            (batch, codebook_size) probabilities
        """
        trans_input = torch.cat([state.h, state.z], dim=-1)
        logits = self.transition_net(trans_input)
        return F.softmax(logits, dim=-1)

    def compute_loss(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute unsupervised loss on observation trajectory.

        Args:
            trajectory: (batch, seq_len, C, H, W) observations

        Returns:
            loss: scalar tensor
            metrics: dict with loss components
        """
        batch_size, seq_len = trajectory.shape[:2]
        device = trajectory.device

        state = self.init_state(batch_size, device)

        total_kl = 0
        total_vq = 0
        total_trans = 0

        for t in range(seq_len - 1):
            obs_t = trajectory[:, t]
            obs_tp1 = trajectory[:, t + 1]

            # Forward pass
            state, outputs = self.forward(obs_t, state)

            total_kl += outputs['kl'].mean()
            total_vq += outputs['vq_loss']

            # Transition prediction loss
            # Get target: posterior code for next timestep
            e_tp1 = self.encoder(obs_tp1)
            post_input_tp1 = torch.cat([state.h, e_tp1], dim=-1)
            post_embedding_tp1 = self.posterior_net(post_input_tp1)
            _, target_idx, _ = self.vq(post_embedding_tp1)

            trans_loss = F.cross_entropy(outputs['transition_logits'], target_idx)
            total_trans += trans_loss

        n_steps = seq_len - 1
        loss = total_kl / n_steps + total_vq / n_steps + total_trans / n_steps

        metrics = {
            'kl': total_kl.item() / n_steps,
            'vq_loss': total_vq.item() / n_steps,
            'transition_loss': total_trans.item() / n_steps,
        }

        return loss, metrics

    def night_update(
        self,
        cumulative_reward: float,
        selection_data: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Night update: compute REINFORCE loss for expert selection.

        The prior network p(code | h) serves as the policy. We reinforce
        code selections that led to high cumulative rewards.

        Args:
            cumulative_reward: total reward from K games (used for logging)
            selection_data: list of tuples from day phase. Supports two formats:
                - (h_state, code_idx) - legacy, uses day-level advantage
                - (h_state, code_idx, game_reward) - per-selection credit assignment

        Returns:
            loss: REINFORCE loss tensor (or None if no selection_data provided)
            metrics: dict with advantage, baseline, loss value
        """
        # Compute REINFORCE loss if selection_data provided
        if selection_data is None or len(selection_data) == 0:
            return None, {'advantage': 0.0, 'baseline': self.baseline.item()}

        # Check format: per-selection rewards or day-level
        has_per_selection_rewards = len(selection_data[0]) == 3

        if has_per_selection_rewards:
            # Per-selection credit assignment
            losses = []
            advantages = []
            log_probs_list = []

            for item in selection_data:
                h_state, code_idx, game_reward = item
                reward_tensor = torch.tensor(game_reward, device=self.baseline.device)

                # Per-selection advantage
                advantage = reward_tensor - self.baseline
                advantages.append(advantage.item())

                # Compute log prob under current prior
                prior_logits = self.prior_net(h_state)
                log_prob = F.log_softmax(prior_logits, dim=-1)
                selected_log_prob = log_prob.gather(1, code_idx.unsqueeze(-1)).squeeze(-1)
                log_probs_list.append(selected_log_prob)

                # Per-selection REINFORCE loss
                losses.append(-advantage.detach() * selected_log_prob)

            # Update baseline with mean reward from all selections
            mean_reward = sum(item[2] for item in selection_data) / len(selection_data)
            alpha = 0.01
            self.baseline = (1 - alpha) * self.baseline + alpha * mean_reward
            self.baseline_count += 1

            reinforce_loss = torch.stack(losses).sum()
            total_log_prob = torch.stack(log_probs_list).sum()

            metrics = {
                'advantage': sum(advantages) / len(advantages),  # Mean advantage
                'baseline': self.baseline.item(),
                'reinforce_loss': reinforce_loss.item(),
                'mean_log_prob': (total_log_prob / len(log_probs_list)).item(),
            }

            return reinforce_loss, metrics

        else:
            # Legacy: day-level advantage (all selections get same signal)
            reward_tensor = torch.tensor(cumulative_reward, device=self.baseline.device)
            advantage = reward_tensor - self.baseline

            # Update baseline
            alpha = 0.01
            self.baseline = (1 - alpha) * self.baseline + alpha * reward_tensor
            self.baseline_count += 1

            log_probs = []
            for h_state, code_idx in selection_data:
                prior_logits = self.prior_net(h_state)
                log_prob = F.log_softmax(prior_logits, dim=-1)
                selected_log_prob = log_prob.gather(1, code_idx.unsqueeze(-1)).squeeze(-1)
                log_probs.append(selected_log_prob)

            total_log_prob = torch.stack(log_probs).sum()
            reinforce_loss = -advantage.detach() * total_log_prob

            metrics = {
                'advantage': advantage.item(),
                'baseline': self.baseline.item(),
                'reinforce_loss': reinforce_loss.item(),
                'mean_log_prob': (total_log_prob / len(log_probs)).item(),
            }

            return reinforce_loss, metrics

    def get_game_embedding(self, state: MetaRSSMState) -> torch.Tensor:
        """Get current game embedding for expert retrieval."""
        return state.z

    def get_game_code(self, state: MetaRSSMState) -> torch.Tensor:
        """Get current game code index."""
        return state.code_idx

    def get_selection_log_prob(
        self,
        state: MetaRSSMState,
        code_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probability of selected code under the prior policy.

        This is used for REINFORCE: the prior p(code | h) serves as
        the policy for expert selection.

        Args:
            state: current MetaRSSMState (need h for prior)
            code_idx: (batch,) selected code indices

        Returns:
            (batch,) log probabilities
        """
        prior_logits = self.prior_net(state.h)
        log_probs = F.log_softmax(prior_logits, dim=-1)
        # Gather log prob for selected codes
        selected_log_probs = log_probs.gather(1, code_idx.unsqueeze(-1)).squeeze(-1)
        return selected_log_probs

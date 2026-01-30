"""
MoE RL Agent: Full model architecture for Atari with tiered experts.

This implements a Decision Transformer-style architecture where:
1. Observations are encoded by a CNN (shared, in HBM)
2. Sequences are processed by transformer blocks with MoE FFN layers
3. Action and value heads produce outputs

The key innovation is that experts are stored across memory tiers
and prefetched based on environment detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, NamedTuple
import math

from .tiered_store import TieredExpertStore
from .moe_layers import TieredMoELayer, MoETransformerBlock
from .env_detector import EnvironmentDetector


class AgentOutput(NamedTuple):
    """Output from the MoE RL agent."""
    action_logits: Tensor      # [batch, num_actions]
    value: Tensor              # [batch, 1]
    expert_ids: List[int]      # Experts used in this forward pass
    aux_loss: Tensor           # Load balancing loss
    env_prediction: int        # Predicted environment
    cache_stats: Dict          # Cache performance stats


class ObservationEncoder(nn.Module):
    """
    CNN encoder for Atari observations.
    
    This is always kept in HBM as it's:
    1. Shared across all games
    2. Small relative to experts
    3. Needed on every forward pass
    
    Architecture follows Nature DQN with some modernizations.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],  # (C, H, W)
        output_dim: int = 512,
        frame_stack: int = 4,
    ):
        super().__init__()
        
        c, h, w = obs_shape
        self.input_channels = c * frame_stack
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, h, w)
            conv_out_size = self.conv(dummy).shape[1]
        
        self.projection = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        
        self.output_dim = output_dim
    
    def forward(self, obs: Tensor) -> Tensor:
        """
        Encode observations.
        
        Args:
            obs: [batch, C*frame_stack, H, W]
            
        Returns:
            features: [batch, output_dim]
        """
        features = self.conv(obs)
        return self.projection(features)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1)]


class MoERLAgent(nn.Module):
    """
    MoE-based RL agent for multi-task Atari.
    
    Architecture:
    1. ObservationEncoder (CNN) - shared, always in HBM
    2. PositionalEncoding - adds temporal structure
    3. MoETransformerBlocks - self-attention + MoE FFN
    4. ActionHead - policy output
    5. ValueHead - value function output
    
    The agent also contains an EnvironmentDetector for prefetching.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        num_experts: int,
        expert_dim: int = 512,
        expert_hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        top_k: int = 2,
        context_len: int = 32,
        frame_stack: int = 4,
        num_envs: int = 57,  # Atari-57
        hbm_capacity: int = 32,
        dram_capacity: int = 128,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        """
        Args:
            obs_shape: Shape of observations (C, H, W)
            num_actions: Number of discrete actions
            num_experts: Total number of experts
            expert_dim: Input/output dimension of experts
            expert_hidden_dim: Hidden dimension within experts
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            top_k: Experts per token
            context_len: Length of context window
            frame_stack: Number of frames to stack
            num_envs: Number of different environments/games
            hbm_capacity: Experts in HBM
            dram_capacity: Experts in DRAM
            dropout: Dropout rate
            device: Compute device
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.context_len = context_len
        self.num_envs = num_envs
        self.device = device
        
        # Create tiered expert store
        self.expert_store = TieredExpertStore(
            num_experts=num_experts,
            expert_dim=expert_dim,
            hidden_dim=expert_hidden_dim,
            hbm_capacity=hbm_capacity,
            dram_capacity=dram_capacity,
            device=device,
            simulate_latency=True,
        )
        
        # Observation encoder (in HBM)
        self.obs_encoder = ObservationEncoder(
            obs_shape=obs_shape,
            output_dim=expert_dim,
            frame_stack=frame_stack,
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(expert_dim, max_len=context_len)
        
        # Token embeddings for action/reward conditioning
        self.action_embedding = nn.Embedding(num_actions, expert_dim)
        self.reward_embedding = nn.Linear(1, expert_dim)
        
        # Transformer blocks with MoE
        self.transformer_blocks = nn.ModuleList([
            MoETransformerBlock(
                dim=expert_dim,
                num_heads=num_heads,
                expert_store=self.expert_store,
                ffn_hidden_dim=expert_hidden_dim,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, num_actions),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, 1),
        )
        
        # Environment detector for prefetching
        self.env_detector = EnvironmentDetector(
            obs_shape=obs_shape,
            num_envs=num_envs,
            num_experts=num_experts,
            hidden_dim=256,
            history_len=frame_stack,
        )
        
        # Track experts used across layers
        self.experts_used: List[int] = []
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in [self.action_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        observations: Tensor,
        actions: Optional[Tensor] = None,
        rewards: Optional[Tensor] = None,
        env_id: Optional[int] = None,
        prefetch: bool = True,
    ) -> AgentOutput:
        """
        Forward pass through the agent.
        
        Args:
            observations: [batch, seq_len, C*frame_stack, H, W]
            actions: Optional past actions [batch, seq_len] (for conditioning)
            rewards: Optional past rewards [batch, seq_len, 1]
            env_id: Current environment ID (for tracking)
            prefetch: Whether to run prefetching
            
        Returns:
            AgentOutput with action logits, value, and auxiliary info
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        
        # Generate context hash for tracking
        context_hash = hash(observations.data_ptr()) if env_id is not None else None
        
        # Run environment detection and prefetching
        env_prediction = -1
        if prefetch:
            # Use last frame for detection
            last_obs = observations[:, -1]
            prefetch_result = self.env_detector.predict(last_obs, top_k=32)
            env_prediction = prefetch_result.predicted_env
            
            # Issue prefetch commands
            self.expert_store.prefetch(prefetch_result.prefetch_set)
        
        # Encode all observations
        obs_flat = observations.reshape(batch_size * seq_len, *observations.shape[2:])
        obs_features = self.obs_encoder(obs_flat)
        obs_features = obs_features.reshape(batch_size, seq_len, -1)
        
        # Build input sequence
        # For Decision Transformer style: interleave (R, s, a) tokens
        # Simplified here: just use observations
        x = obs_features
        
        # Add action/reward conditioning if provided
        if actions is not None:
            action_emb = self.action_embedding(actions)
            x = x + action_emb
        
        if rewards is not None:
            reward_emb = self.reward_embedding(rewards)
            x = x + reward_emb
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal attention mask
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Forward through transformer blocks
        total_aux_loss = 0.0
        self.experts_used = []
        
        for block in self.transformer_blocks:
            x, aux_loss = block(x, attention_mask=causal_mask, context_hash=context_hash)
            total_aux_loss = total_aux_loss + aux_loss
            
            # Track experts used
            routing_stats = block.get_last_routing_stats()
            if 'expert_usage' in routing_stats:
                self.experts_used.extend(routing_stats['expert_usage'].keys())
        
        # Get final representation (last position)
        final_repr = x[:, -1]  # [batch, dim]
        
        # Compute outputs
        action_logits = self.action_head(final_repr)
        value = self.value_head(final_repr)
        
        # Get cache stats
        cache_stats = self.expert_store.get_stats_summary()
        
        return AgentOutput(
            action_logits=action_logits,
            value=value,
            expert_ids=list(set(self.experts_used)),
            aux_loss=total_aux_loss,
            env_prediction=env_prediction,
            cache_stats=cache_stats,
        )
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask
    
    @torch.no_grad()
    def get_action(
        self,
        observations: Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """
        Get action for a single observation.
        
        Args:
            observations: [1, seq_len, C*frame_stack, H, W]
            deterministic: Whether to use argmax or sample
            
        Returns:
            action: Selected action
            value: Value estimate
        """
        self.eval()
        
        output = self.forward(observations, prefetch=True)
        
        if deterministic:
            action = output.action_logits.argmax(dim=-1).item()
        else:
            probs = F.softmax(output.action_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        
        return action, output.value.item()
    
    def get_expert_specialization(self) -> Dict[int, Dict[int, int]]:
        """
        Analyze which experts specialize in which environments.
        
        Returns:
            {env_id -> {expert_id -> usage_count}}
        """
        # This would be populated during training
        # Placeholder for now
        return {}
    
    def save_checkpoint(self, path: str):
        """Save full model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'expert_store': {
                'nvme_store': self.expert_store.nvme_store,
                'stats': self.expert_store.stats,
            },
            'env_detector_state_dict': self.env_detector.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.expert_store.nvme_store = checkpoint['expert_store']['nvme_store']
        self.expert_store.stats = checkpoint['expert_store']['stats']
        self.env_detector.load_state_dict(checkpoint['env_detector_state_dict'])


class MoERLAgentConfig:
    """Configuration for MoE RL Agent."""
    
    def __init__(
        self,
        # Environment
        obs_shape: Tuple[int, ...] = (1, 84, 84),
        num_actions: int = 18,
        num_envs: int = 57,
        frame_stack: int = 4,
        
        # Model architecture
        num_experts: int = 256,
        expert_dim: int = 512,
        expert_hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        top_k: int = 2,
        context_len: int = 32,
        dropout: float = 0.1,
        
        # Memory hierarchy
        hbm_capacity: int = 32,
        dram_capacity: int = 128,
        
        # Training
        learning_rate: float = 3e-4,
        aux_loss_weight: float = 0.01,
    ):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.frame_stack = frame_stack
        
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.top_k = top_k
        self.context_len = context_len
        self.dropout = dropout
        
        self.hbm_capacity = hbm_capacity
        self.dram_capacity = dram_capacity
        
        self.learning_rate = learning_rate
        self.aux_loss_weight = aux_loss_weight
    
    def create_agent(self, device: str = "cuda") -> MoERLAgent:
        """Create agent from config."""
        return MoERLAgent(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            num_experts=self.num_experts,
            expert_dim=self.expert_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            top_k=self.top_k,
            context_len=self.context_len,
            frame_stack=self.frame_stack,
            num_envs=self.num_envs,
            hbm_capacity=self.hbm_capacity,
            dram_capacity=self.dram_capacity,
            dropout=self.dropout,
            device=device,
        )
    
    def estimate_parameter_count(self) -> Dict[str, int]:
        """Estimate parameter counts for different components."""
        # Expert parameters
        expert_params = self.num_experts * (
            self.expert_dim * self.expert_hidden_dim +  # w1
            self.expert_hidden_dim * self.expert_dim +  # w2
            self.expert_hidden_dim +                    # b1
            self.expert_dim                             # b2
        )
        
        # Per-layer non-expert params (attention)
        attention_params = self.num_layers * (
            4 * self.expert_dim * self.expert_dim  # Q, K, V, O projections
        )
        
        # Encoder
        encoder_params = 32 * 8 * 8 + 32 * 64 * 4 * 4 + 64 * 64 * 3 * 3
        
        # Heads
        head_params = 2 * (self.expert_dim * self.expert_dim + self.expert_dim * self.num_actions)
        
        return {
            'expert_params': expert_params,
            'attention_params': attention_params,
            'encoder_params': encoder_params,
            'head_params': head_params,
            'total': expert_params + attention_params + encoder_params + head_params,
            'params_in_hbm': attention_params + encoder_params + head_params + 
                            self.hbm_capacity * (self.expert_dim * self.expert_hidden_dim * 2),
            'params_in_dram': self.dram_capacity * (self.expert_dim * self.expert_hidden_dim * 2),
        }

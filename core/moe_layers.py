"""
Mixture of Experts layers with tiered storage integration.

Key design decisions:
1. Router is always in HBM (tiny, always needed)
2. Experts are fetched on-demand from tiered storage
3. Load balancing loss encourages diverse expert usage
4. Auxiliary losses for router training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, NamedTuple
import math

from .tiered_store import TieredExpertStore


class RouterOutput(NamedTuple):
    """Output from the router network."""
    expert_indices: Tensor  # [batch, top_k] - selected expert IDs
    expert_weights: Tensor  # [batch, top_k] - weights for combining experts
    router_logits: Tensor   # [batch, num_experts] - raw logits for aux loss
    load_balancing_loss: Tensor  # Scalar loss for load balancing


class ExpertRouter(nn.Module):
    """
    Router network that selects experts for each input.
    
    This is always kept in HBM as it's small and accessed on every forward pass.
    Implements top-k routing with load balancing.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_experts: Total number of experts to route to
            top_k: Number of experts to select per input
            noise_std: Std of noise added during training for exploration
            capacity_factor: Multiplier for expert capacity (for load balancing)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor
        
        # Router is a simple linear projection
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize with small weights for stable routing initially
        nn.init.normal_(self.router.weight, std=0.01)
    
    def forward(
        self, 
        x: Tensor,
        training: bool = True
    ) -> RouterOutput:
        """
        Route inputs to experts.
        
        Args:
            x: Input tensor of shape [batch, dim]
            training: Whether we're in training mode (adds noise)
            
        Returns:
            RouterOutput with selected experts and weights
        """
        batch_size = x.shape[0]
        
        # Compute router logits
        router_logits = self.router(x)  # [batch, num_experts]
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            noisy_logits = router_logits + noise
        else:
            noisy_logits = router_logits
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(
            noisy_logits, self.top_k, dim=-1
        )  # [batch, top_k]
        
        # Softmax over selected experts for combination weights
        expert_weights = F.softmax(top_k_logits, dim=-1)  # [batch, top_k]
        
        # Compute load balancing loss
        # This encourages uniform expert utilization
        load_balancing_loss = self._compute_load_balancing_loss(
            router_logits, top_k_indices, batch_size
        )
        
        return RouterOutput(
            expert_indices=top_k_indices,
            expert_weights=expert_weights,
            router_logits=router_logits,
            load_balancing_loss=load_balancing_loss
        )
    
    def _compute_load_balancing_loss(
        self,
        router_logits: Tensor,
        selected_experts: Tensor,
        batch_size: int
    ) -> Tensor:
        """
        Compute auxiliary loss for load balancing.
        
        From Switch Transformer paper: encourages routing decisions
        to be spread across experts rather than collapsing to few.
        """
        # Fraction of tokens routed to each expert
        # selected_experts: [batch, top_k]
        expert_mask = F.one_hot(selected_experts, self.num_experts).float()
        # [batch, top_k, num_experts] -> [batch, num_experts]
        expert_mask = expert_mask.sum(dim=1)
        tokens_per_expert = expert_mask.sum(dim=0)  # [num_experts]
        fraction_tokens = tokens_per_expert / batch_size
        
        # Router probability for each expert
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, num_experts]
        mean_router_prob = router_probs.mean(dim=0)  # [num_experts]
        
        # Load balancing loss: dot product of fractions and probs
        # Minimized when both are uniform
        load_balance_loss = (
            self.num_experts * (fraction_tokens * mean_router_prob).sum()
        )
        
        return load_balance_loss
    
    def get_expert_utilization(self, expert_indices: Tensor) -> Dict[int, int]:
        """Get count of how many times each expert was selected."""
        counts = {}
        for idx in expert_indices.flatten().tolist():
            counts[idx] = counts.get(idx, 0) + 1
        return counts


class TieredMoELayer(nn.Module):
    """
    Mixture of Experts layer with tiered storage backend.
    
    This is the main module that combines:
    1. Router (in HBM) - decides which experts to use
    2. TieredExpertStore - manages expert parameters across memory tiers
    3. Expert computation - fetches and applies experts
    
    Key insight: The router learns to predict expert needs, which
    enables the prefetching system to work effectively.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        expert_store: TieredExpertStore,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input/output dimension
            hidden_dim: Hidden dimension within experts
            expert_store: TieredExpertStore managing expert parameters
            top_k: Number of experts per input
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expert_store = expert_store
        self.top_k = top_k
        
        # Router is always in HBM
        self.router = ExpertRouter(
            input_dim=input_dim,
            num_experts=expert_store.num_experts,
            top_k=top_k,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Track which experts were used (for analysis)
        self.last_expert_indices: Optional[Tensor] = None
        self.last_expert_weights: Optional[Tensor] = None
    
    def forward(
        self, 
        x: Tensor,
        context_hash: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, dim]
            context_hash: Optional context identifier for access tracking
            
        Returns:
            output: Transformed tensor (same shape as input)
            aux_loss: Load balancing loss for training
        """
        # Handle both [batch, dim] and [batch, seq, dim]
        input_shape = x.shape
        if len(input_shape) == 3:
            batch, seq_len, dim = input_shape
            x = x.reshape(batch * seq_len, dim)
        else:
            seq_len = None
        
        # Normalize input
        x_norm = self.layer_norm(x)
        
        # Route to experts
        router_output = self.router(x_norm, training=self.training)
        
        # Store for analysis
        self.last_expert_indices = router_output.expert_indices
        self.last_expert_weights = router_output.expert_weights
        
        # Get unique experts needed
        unique_experts = router_output.expert_indices.unique().tolist()
        
        # Fetch expert parameters from tiered storage
        expert_params = self.expert_store.get_experts(
            unique_experts, 
            context_hash=context_hash
        )
        
        # Apply experts and combine outputs
        output = self._apply_experts(
            x_norm,
            router_output.expert_indices,
            router_output.expert_weights,
            expert_params
        )
        
        # Residual connection and dropout
        output = x + self.dropout(output)
        
        # Reshape back if needed
        if seq_len is not None:
            output = output.reshape(batch, seq_len, dim)
        
        return output, router_output.load_balancing_loss
    
    def _apply_experts(
        self,
        x: Tensor,
        expert_indices: Tensor,
        expert_weights: Tensor,
        expert_params: Dict[int, Dict[str, Tensor]]
    ) -> Tensor:
        """
        Apply selected experts to inputs and combine results.
        
        This is where the actual expert computation happens.
        """
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        # Process each expert
        for k in range(self.top_k):
            # Get expert indices and weights for this slot
            indices = expert_indices[:, k]  # [batch]
            weights = expert_weights[:, k]  # [batch]
            
            # Group inputs by expert for efficient batched computation
            for expert_id in indices.unique().tolist():
                # Find inputs routed to this expert
                mask = (indices == expert_id)
                if not mask.any():
                    continue
                
                # Get expert parameters
                params = expert_params[expert_id]
                
                # Apply expert: x @ W1.T + b1 -> relu -> @ W2.T + b2
                x_expert = x[mask]
                h = F.relu(x_expert @ params['w1'].T + params['b1'])
                y = h @ params['w2'].T + params['b2']
                
                # Weight by routing score
                output[mask] += weights[mask].unsqueeze(-1) * y
        
        return output
    
    def get_routing_stats(self) -> Dict:
        """Get statistics about routing decisions."""
        if self.last_expert_indices is None:
            return {}
        
        indices = self.last_expert_indices.flatten()
        unique, counts = torch.unique(indices, return_counts=True)
        
        return {
            'num_unique_experts': len(unique),
            'expert_usage': {int(e): int(c) for e, c in zip(unique, counts)},
            'top_expert': int(unique[counts.argmax()]),
            'entropy': self._compute_routing_entropy(),
        }
    
    def _compute_routing_entropy(self) -> float:
        """Compute entropy of routing distribution."""
        if self.last_expert_indices is None:
            return 0.0
        
        indices = self.last_expert_indices.flatten()
        counts = torch.bincount(indices, minlength=self.expert_store.num_experts)
        probs = counts.float() / counts.sum()
        probs = probs[probs > 0]  # Avoid log(0)
        entropy = -(probs * probs.log()).sum()
        return float(entropy)


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE feedforward layer.
    
    Structure:
    1. Self-attention (dense, in HBM)
    2. MoE FFN (sparse, tiered storage)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expert_store: TieredExpertStore,
        ffn_hidden_dim: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        
        # Self-attention (always in HBM)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        # MoE feedforward
        self.moe = TieredMoELayer(
            input_dim=dim,
            hidden_dim=ffn_hidden_dim,
            expert_store=expert_store,
            top_k=top_k,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        context_hash: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input [batch, seq_len, dim]
            attention_mask: Optional attention mask
            context_hash: Context identifier for expert tracking
            
        Returns:
            output: Transformed tensor
            aux_loss: Load balancing loss
        """
        # Self-attention with residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            attn_mask=attention_mask,
            need_weights=False
        )
        x = x + self.attn_dropout(attn_out)
        
        # MoE feedforward with residual (handled inside TieredMoELayer)
        x, aux_loss = self.moe(x, context_hash=context_hash)
        
        return x, aux_loss
    
    def get_last_routing_stats(self) -> Dict:
        """Get routing stats from last forward pass."""
        return self.moe.get_routing_stats()

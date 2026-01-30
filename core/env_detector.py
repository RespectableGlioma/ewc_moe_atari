"""
Environment Detector: Predicts which experts will be needed based on observations.

This is the key module that enables effective prefetching. It learns:
1. Which environment/game we're currently in
2. Which experts are likely to be needed
3. Temporal patterns in expert access

The detector is intentionally small so it can run ahead of the main
computation and issue prefetch commands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
from collections import deque


class PredictionOutput(NamedTuple):
    """Output from the environment detector."""
    env_logits: Tensor          # [batch, num_envs] - environment classification
    expert_probs: Tensor        # [batch, num_experts] - probability each expert needed
    prefetch_set: List[int]     # Expert IDs to prefetch
    predicted_env: int          # Most likely environment ID
    confidence: float           # Confidence in environment prediction


class EnvironmentDetector(nn.Module):
    """
    Lightweight network that predicts expert needs from observations.
    
    Key design principles:
    1. Very small (runs on every step with minimal overhead)
    2. Processes recent frames to detect environment
    3. Outputs probability distribution over experts
    4. Learns from actual expert access patterns
    
    The detector essentially learns a compressed representation of the
    mapping from observations -> which experts will be useful.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],  # (C, H, W) for Atari
        num_envs: int,
        num_experts: int,
        hidden_dim: int = 256,
        history_len: int = 4,  # Number of frames to consider
    ):
        """
        Args:
            obs_shape: Shape of observations (C, H, W)
            num_envs: Number of possible environments/games
            num_experts: Total number of experts in the system
            hidden_dim: Hidden dimension for the detector
            history_len: Number of recent frames to use
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.num_envs = num_envs
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        
        # Small CNN encoder
        # Assumes 84x84 Atari frames, but adaptable
        c, h, w = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c * history_len, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute encoder output size
        with torch.no_grad():
            dummy = torch.zeros(1, c * history_len, h, w)
            encoder_out_size = self.encoder(dummy).shape[1]
        
        # Projection to hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Environment classification head
        self.env_classifier = nn.Linear(hidden_dim, num_envs)
        
        # Expert prediction head
        # Outputs probability that each expert will be needed
        self.expert_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Running statistics for adaptive thresholding
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self.ema_decay = 0.99
        
        # Frame buffer for history
        self.frame_buffer = None
    
    def forward(
        self,
        obs: Tensor,
        return_hidden: bool = False
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward pass through detector.
        
        Args:
            obs: Observation tensor [batch, C*history, H, W] or [batch, C, H, W]
            return_hidden: Whether to return hidden representation
            
        Returns:
            env_logits: [batch, num_envs]
            expert_probs: [batch, num_experts]
            hidden: [batch, hidden_dim] if return_hidden else None
        """
        # Encode observation
        features = self.encoder(obs)
        hidden = self.projection(features)
        
        # Classify environment
        env_logits = self.env_classifier(hidden)
        
        # Predict expert probabilities
        expert_logits = self.expert_predictor(hidden)
        expert_probs = torch.sigmoid(expert_logits)
        
        if return_hidden:
            return env_logits, expert_probs, hidden
        return env_logits, expert_probs, None
    
    @torch.no_grad()
    def predict(
        self,
        obs: Tensor,
        top_k: int = 32,
        threshold: float = 0.3,
    ) -> PredictionOutput:
        """
        Make predictions for prefetching.
        
        Args:
            obs: Current observation(s)
            top_k: Maximum number of experts to prefetch
            threshold: Minimum probability to include in prefetch set
            
        Returns:
            PredictionOutput with prefetch recommendations
        """
        self.eval()
        
        env_logits, expert_probs, _ = self.forward(obs)
        
        # Get environment prediction
        env_probs = F.softmax(env_logits, dim=-1)
        predicted_env = env_probs.argmax(dim=-1).item()
        confidence = env_probs.max(dim=-1).values.item()
        
        # Get experts to prefetch
        # Use both top-k and threshold
        probs = expert_probs.squeeze(0)  # Assume batch=1 for prefetching
        
        # Adaptive threshold based on running statistics
        adaptive_threshold = min(threshold, probs.mean().item() + probs.std().item())
        
        above_threshold = (probs > adaptive_threshold).nonzero(as_tuple=True)[0]
        top_k_indices = probs.topk(min(top_k, len(probs))).indices
        
        # Union of threshold-based and top-k
        prefetch_set = list(set(above_threshold.tolist()) | set(top_k_indices.tolist()))
        prefetch_set = prefetch_set[:top_k]  # Limit to top_k
        
        return PredictionOutput(
            env_logits=env_logits,
            expert_probs=expert_probs,
            prefetch_set=prefetch_set,
            predicted_env=predicted_env,
            confidence=confidence,
        )
    
    def update_usage_stats(self, expert_ids: List[int]):
        """Update running statistics of expert usage."""
        usage = torch.zeros(self.num_experts, device=self.expert_usage_ema.device)
        for eid in expert_ids:
            usage[eid] += 1
        usage = usage / max(len(expert_ids), 1)
        
        self.expert_usage_ema = (
            self.ema_decay * self.expert_usage_ema + 
            (1 - self.ema_decay) * usage
        )
    
    def get_prefetch_set(
        self,
        obs: Tensor,
        top_k: int = 32
    ) -> List[int]:
        """Convenience method to just get prefetch set."""
        prediction = self.predict(obs, top_k=top_k)
        return prediction.prefetch_set


class EnvironmentDetectorTrainer:
    """
    Trainer for the environment detector.
    
    The detector is trained with two objectives:
    1. Environment classification (supervised from env labels)
    2. Expert prediction (supervised from actual routing decisions)
    
    This enables the detector to learn the mapping from observations
    to expert needs, which is what makes prefetching possible.
    """
    
    def __init__(
        self,
        detector: EnvironmentDetector,
        lr: float = 1e-4,
        env_loss_weight: float = 1.0,
        expert_loss_weight: float = 1.0,
    ):
        self.detector = detector
        self.optimizer = torch.optim.Adam(detector.parameters(), lr=lr)
        self.env_loss_weight = env_loss_weight
        self.expert_loss_weight = expert_loss_weight
        
        # Replay buffer for training samples
        self.buffer = deque(maxlen=10000)
    
    def store_sample(
        self,
        obs: Tensor,
        env_id: int,
        accessed_experts: List[int]
    ):
        """Store a training sample."""
        # Create expert target (multi-hot)
        expert_target = torch.zeros(self.detector.num_experts)
        for eid in accessed_experts:
            expert_target[eid] = 1.0
        
        self.buffer.append({
            'obs': obs.cpu(),
            'env_id': env_id,
            'expert_target': expert_target,
        })
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Perform one training step on the detector.
        
        Returns:
            Dictionary of loss values
        """
        if len(self.buffer) < batch_size:
            return {'env_loss': 0.0, 'expert_loss': 0.0, 'total_loss': 0.0}
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Stack into tensors
        obs = torch.stack([b['obs'] for b in batch])
        env_ids = torch.tensor([b['env_id'] for b in batch])
        expert_targets = torch.stack([b['expert_target'] for b in batch])
        
        # Move to device
        device = next(self.detector.parameters()).device
        obs = obs.to(device)
        env_ids = env_ids.to(device)
        expert_targets = expert_targets.to(device)
        
        # Forward pass
        self.detector.train()
        env_logits, expert_probs, _ = self.detector(obs)
        
        # Environment classification loss
        env_loss = F.cross_entropy(env_logits, env_ids)
        
        # Expert prediction loss (binary cross-entropy)
        expert_loss = F.binary_cross_entropy(expert_probs, expert_targets)
        
        # Combined loss
        total_loss = (
            self.env_loss_weight * env_loss +
            self.expert_loss_weight * expert_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'env_loss': env_loss.item(),
            'expert_loss': expert_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def compute_prefetch_accuracy(self, recent_n: int = 100) -> Dict[str, float]:
        """
        Compute how well prefetch predictions match actual access.
        
        This is a key metric for evaluating the detector.
        """
        if len(self.buffer) < recent_n:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        recent = list(self.buffer)[-recent_n:]
        
        total_precision = 0.0
        total_recall = 0.0
        
        self.detector.eval()
        device = next(self.detector.parameters()).device
        
        with torch.no_grad():
            for sample in recent:
                obs = sample['obs'].unsqueeze(0).to(device)
                prediction = self.detector.predict(obs, top_k=32)
                
                predicted_set = set(prediction.prefetch_set)
                actual_set = set(
                    i for i, v in enumerate(sample['expert_target']) if v > 0.5
                )
                
                if predicted_set:
                    precision = len(predicted_set & actual_set) / len(predicted_set)
                else:
                    precision = 0.0
                    
                if actual_set:
                    recall = len(predicted_set & actual_set) / len(actual_set)
                else:
                    recall = 1.0
                
                total_precision += precision
                total_recall += recall
        
        avg_precision = total_precision / recent_n
        avg_recall = total_recall / recent_n
        
        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1,
        }


class HierarchicalEnvironmentDetector(nn.Module):
    """
    Two-level detector: coarse environment, then fine-grained expert prediction.
    
    Level 1: Classify into broad environment categories
    Level 2: Within each category, predict specific experts
    
    This hierarchical approach can be more efficient when there are
    many environments and experts, as it narrows down the search space.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        num_env_categories: int,  # Coarse categories (e.g., game genres)
        num_envs: int,            # Specific environments
        num_experts: int,
        experts_per_category: int,  # Experts to consider per category
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.num_env_categories = num_env_categories
        self.num_envs = num_envs
        self.num_experts = num_experts
        self.experts_per_category = experts_per_category
        
        # Shared encoder
        c, h, w = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c * 4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, c * 4, h, w)
            encoder_out_size = self.encoder(dummy).shape[1]
        
        self.projection = nn.Linear(encoder_out_size, hidden_dim)
        
        # Level 1: Coarse category classifier
        self.category_classifier = nn.Linear(hidden_dim, num_env_categories)
        
        # Level 2: Per-category expert predictors
        # Each category has its own expert prediction head
        self.category_expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, experts_per_category)
            )
            for _ in range(num_env_categories)
        ])
        
        # Mapping from (category, local_expert_idx) -> global_expert_id
        # This is learned/updated during training
        self.register_buffer(
            'category_expert_mapping',
            torch.zeros(num_env_categories, experts_per_category, dtype=torch.long)
        )
        self._initialize_random_mapping()
    
    def _initialize_random_mapping(self):
        """Initialize random category-to-expert mapping."""
        for cat in range(self.num_env_categories):
            experts = torch.randperm(self.num_experts)[:self.experts_per_category]
            self.category_expert_mapping[cat] = experts
    
    def update_mapping_from_usage(self, category_expert_counts: Dict[int, Dict[int, int]]):
        """
        Update category-expert mapping based on observed usage patterns.
        
        Args:
            category_expert_counts: {category -> {expert_id -> count}}
        """
        for cat, expert_counts in category_expert_counts.items():
            # Get top experts for this category
            sorted_experts = sorted(
                expert_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.experts_per_category]
            
            new_experts = torch.tensor(
                [e for e, _ in sorted_experts],
                dtype=torch.long,
                device=self.category_expert_mapping.device
            )
            
            # Pad if needed
            if len(new_experts) < self.experts_per_category:
                padding = torch.randint(
                    0, self.num_experts,
                    (self.experts_per_category - len(new_experts),),
                    device=self.category_expert_mapping.device
                )
                new_experts = torch.cat([new_experts, padding])
            
            self.category_expert_mapping[cat] = new_experts
    
    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.
        
        Returns:
            category_logits: [batch, num_categories]
            expert_probs: [batch, num_experts] (sparse)
            hidden: [batch, hidden_dim]
        """
        features = self.encoder(obs)
        hidden = F.relu(self.projection(features))
        
        # Level 1: Category prediction
        category_logits = self.category_classifier(hidden)
        category_probs = F.softmax(category_logits, dim=-1)
        
        # Level 2: Per-category expert prediction
        batch_size = obs.shape[0]
        expert_probs = torch.zeros(
            batch_size, self.num_experts, 
            device=obs.device
        )
        
        for cat in range(self.num_env_categories):
            # Get local expert probs for this category
            local_probs = torch.sigmoid(self.category_expert_heads[cat](hidden))
            
            # Weight by category probability
            weighted_probs = category_probs[:, cat:cat+1] * local_probs
            
            # Map to global expert IDs
            global_ids = self.category_expert_mapping[cat]
            expert_probs.scatter_add_(
                1, 
                global_ids.unsqueeze(0).expand(batch_size, -1),
                weighted_probs
            )
        
        return category_logits, expert_probs, hidden
    
    @torch.no_grad()
    def get_prefetch_set(self, obs: Tensor, top_k: int = 32) -> List[int]:
        """Get experts to prefetch."""
        self.eval()
        _, expert_probs, _ = self.forward(obs)
        top_experts = expert_probs.squeeze(0).topk(top_k).indices
        return top_experts.tolist()

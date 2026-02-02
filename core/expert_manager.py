"""
ExpertManager: Orchestrates save/load/create logic for experts.

Manages the expert library with epsilon-threshold for novelty detection.
When a new game embedding is far from all stored experts, creates a new one.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
import logging

from .expert import Expert
from .tiered_store import TieredStore

logger = logging.getLogger(__name__)


class ExpertManager:
    """
    Manages expert lifecycle: retrieval, creation, saving, loading.

    Uses embedding similarity to determine which expert to load.
    Creates new experts when no existing expert is within epsilon threshold.
    """

    def __init__(
        self,
        storage_path: str = "./expert_library",
        device: torch.device = torch.device("cuda"),
        epsilon: float = 0.5,  # Similarity threshold for novelty
        code_dim: int = 32,
        codebook_size: int = 64,
        num_actions: int = 18,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        max_experts_in_memory: int = 2,  # Active + prefetched
    ):
        self.device = device
        self.epsilon = epsilon
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.max_experts_in_memory = max_experts_in_memory

        # Tiered storage for experts
        self.storage = TieredStore(storage_path)

        # Expert registry: code_idx -> expert_id mapping
        self.code_to_expert: Dict[int, str] = {}

        # Embedding centroids for each expert (for similarity matching)
        # expert_id -> centroid embedding
        self.expert_centroids: Dict[str, torch.Tensor] = {}

        # Currently loaded experts
        self.active_expert: Optional[Expert] = None
        self.active_expert_id: Optional[str] = None
        self.prefetched_expert: Optional[Expert] = None
        self.prefetched_expert_id: Optional[str] = None

        # Statistics
        self.stats = {
            'total_experts_created': 0,
            'total_swaps': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

        # Load existing registry if present
        self._load_registry()

    def _load_registry(self):
        """Load expert registry from storage."""
        registry = self.storage.load_registry()
        if registry:
            self.code_to_expert = registry.get('code_to_expert', {})
            # Convert string keys back to int
            self.code_to_expert = {
                int(k): v for k, v in self.code_to_expert.items()
            }
            # Load centroids
            centroids_data = registry.get('expert_centroids', {})
            for expert_id, centroid_list in centroids_data.items():
                self.expert_centroids[expert_id] = torch.tensor(
                    centroid_list, device=self.device
                )
            logger.info(f"Loaded registry with {len(self.code_to_expert)} experts")

    def _save_registry(self):
        """Save expert registry to storage."""
        registry = {
            'code_to_expert': {str(k): v for k, v in self.code_to_expert.items()},
            'expert_centroids': {
                k: v.cpu().tolist() for k, v in self.expert_centroids.items()
            },
        }
        self.storage.save_registry(registry)

    def create_expert(self, expert_id: Optional[str] = None) -> Expert:
        """Create a new expert with fresh weights."""
        if expert_id is None:
            expert_id = f"expert_{self.stats['total_experts_created']:04d}"

        expert = Expert(
            obs_shape=self.obs_shape,
            num_actions=self.num_actions,
            expert_id=expert_id,
        )
        expert.metadata['creation_time'] = time.time()
        expert.to(self.device)

        self.stats['total_experts_created'] += 1
        logger.info(f"Created new expert: {expert_id}")

        return expert

    def find_similar_expert(
        self,
        embedding: torch.Tensor,
    ) -> Tuple[Optional[str], float]:
        """
        Find the most similar expert to the given embedding.

        Args:
            embedding: (code_dim,) game embedding from meta-agent

        Returns:
            expert_id: ID of most similar expert, or None if none within epsilon
            similarity: cosine similarity score
        """
        if not self.expert_centroids:
            return None, 0.0

        embedding = embedding.detach().view(1, -1)

        best_expert_id = None
        best_similarity = -float('inf')

        for expert_id, centroid in self.expert_centroids.items():
            centroid = centroid.view(1, -1)
            similarity = F.cosine_similarity(embedding, centroid).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_expert_id = expert_id

        # Check if within epsilon threshold
        if best_similarity < (1 - self.epsilon):
            return None, best_similarity

        return best_expert_id, best_similarity

    def retrieve_or_create(
        self,
        embedding: torch.Tensor,
        code_idx: Optional[int] = None,
    ) -> Expert:
        """
        Retrieve existing expert or create new one based on embedding.

        This is the main entry point for expert management during gameplay.

        Args:
            embedding: (code_dim,) game embedding from meta-agent
            code_idx: optional discrete code index

        Returns:
            Expert to use for this game
        """
        # First check code-based lookup if available
        if code_idx is not None and code_idx in self.code_to_expert:
            expert_id = self.code_to_expert[code_idx]

            # Check if already loaded
            if expert_id == self.active_expert_id:
                self.stats['cache_hits'] += 1
                return self.active_expert

            if expert_id == self.prefetched_expert_id:
                self.stats['cache_hits'] += 1
                # Swap prefetched to active
                self._swap_prefetched_to_active()
                return self.active_expert

            # Load from storage
            self.stats['cache_misses'] += 1
            return self._load_expert(expert_id)

        # Embedding-based similarity search
        expert_id, similarity = self.find_similar_expert(embedding)

        if expert_id is not None:
            logger.info(f"Found similar expert {expert_id} (sim={similarity:.3f})")

            if expert_id == self.active_expert_id:
                self.stats['cache_hits'] += 1
                return self.active_expert

            if expert_id == self.prefetched_expert_id:
                self.stats['cache_hits'] += 1
                self._swap_prefetched_to_active()
                return self.active_expert

            self.stats['cache_misses'] += 1
            return self._load_expert(expert_id)

        # No similar expert found - create new one
        logger.info(f"No similar expert found (best sim={similarity:.3f}), creating new")
        expert = self.create_expert()

        # Register embedding as centroid for new expert
        self.expert_centroids[expert.expert_id] = embedding.detach().clone()

        # Register code mapping if available
        if code_idx is not None:
            self.code_to_expert[code_idx] = expert.expert_id

        # Set as active
        self._save_current_expert()
        self.active_expert = expert
        self.active_expert_id = expert.expert_id

        self._save_registry()

        return expert

    def _load_expert(self, expert_id: str) -> Expert:
        """Load expert from storage."""
        # Save current expert first
        self._save_current_expert()

        # Load requested expert
        data = self.storage.load_expert(expert_id)
        if data is None:
            logger.warning(f"Expert {expert_id} not found, creating new")
            expert = self.create_expert(expert_id)
        else:
            expert = Expert.from_state_dict_with_metadata(data, self.device)
            logger.info(f"Loaded expert {expert_id} from storage")

        self.active_expert = expert
        self.active_expert_id = expert_id
        self.stats['total_swaps'] += 1

        return expert

    def _save_current_expert(self):
        """Save currently active expert to storage."""
        if self.active_expert is not None:
            data = self.active_expert.get_state_dict_with_metadata()
            self.storage.save_expert(self.active_expert_id, data)
            logger.debug(f"Saved expert {self.active_expert_id}")

    def _swap_prefetched_to_active(self):
        """Swap prefetched expert to active slot."""
        self._save_current_expert()

        self.active_expert = self.prefetched_expert
        self.active_expert_id = self.prefetched_expert_id
        self.prefetched_expert = None
        self.prefetched_expert_id = None
        self.stats['total_swaps'] += 1

    def prefetch_expert(self, expert_id: str):
        """
        Prefetch an expert in anticipation of needing it.

        Called when meta-agent predicts upcoming game transition.
        """
        if expert_id == self.active_expert_id:
            return  # Already active

        if expert_id == self.prefetched_expert_id:
            return  # Already prefetched

        # Load into prefetch slot
        data = self.storage.load_expert(expert_id)
        if data is not None:
            self.prefetched_expert = Expert.from_state_dict_with_metadata(
                data, self.device
            )
            self.prefetched_expert_id = expert_id
            logger.debug(f"Prefetched expert {expert_id}")

    def prefetch_from_distribution(
        self,
        transition_probs: torch.Tensor,
        threshold: float = 0.3,
    ):
        """
        Prefetch most likely next expert based on transition distribution.

        Args:
            transition_probs: (codebook_size,) probabilities over next codes
            threshold: minimum probability to trigger prefetch
        """
        probs, indices = transition_probs.sort(descending=True)

        for prob, code_idx in zip(probs[:3], indices[:3]):
            if prob < threshold:
                break

            code_idx = code_idx.item()
            if code_idx in self.code_to_expert:
                expert_id = self.code_to_expert[code_idx]
                if expert_id != self.active_expert_id:
                    self.prefetch_expert(expert_id)
                    return

    def update_centroid(
        self,
        expert_id: str,
        embedding: torch.Tensor,
        alpha: float = 0.1,
    ):
        """
        Update expert centroid with exponential moving average.

        Called during training to keep centroids up to date.
        """
        if expert_id in self.expert_centroids:
            old_centroid = self.expert_centroids[expert_id]
            new_centroid = (1 - alpha) * old_centroid + alpha * embedding.detach()
            self.expert_centroids[expert_id] = new_centroid
        else:
            self.expert_centroids[expert_id] = embedding.detach().clone()

    def register_code_mapping(self, code_idx: int, expert_id: str):
        """Register a mapping from code index to expert."""
        self.code_to_expert[code_idx] = expert_id
        self._save_registry()

    def get_active_expert(self) -> Optional[Expert]:
        """Get currently active expert."""
        return self.active_expert

    def get_stats(self) -> dict:
        """Get manager statistics."""
        return {
            **self.stats,
            'num_experts_stored': len(self.expert_centroids),
            'active_expert': self.active_expert_id,
            'prefetched_expert': self.prefetched_expert_id,
        }

    def save_all(self):
        """Save all state to storage."""
        self._save_current_expert()
        self._save_registry()
        logger.info("Saved all expert manager state")

    def list_experts(self) -> List[str]:
        """List all stored expert IDs."""
        return list(self.expert_centroids.keys())

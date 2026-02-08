"""
ExpertManager: Orchestrates save/load/create logic for experts.

Manages the expert library with epsilon-threshold for novelty detection.
When a new game embedding is far from all stored experts, creates a new one.

Uses reward-weighted code→expert affinity for learned mappings that become
"sticky" based on proven performance over time.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from collections import defaultdict
import math
import time
import logging
import random

from .expert import Expert
from .tiered_store import TieredStore

logger = logging.getLogger(__name__)


class CodeExpertAffinity:
    """
    Tracks reward-weighted affinity between VQ codes and experts.

    High-performing (code, expert) pairs become "sticky" over time,
    while poor performers can be displaced by better matches.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,        # EMA decay for reward tracking
        stickiness_scale: float = 1.0,  # How much visit count matters
        exploration_rate: float = 0.1,  # Probability of exploring alternatives
        min_score_threshold: float = 0.0,  # Minimum score to use affinity
    ):
        self.ema_alpha = ema_alpha
        self.stickiness_scale = stickiness_scale
        self.exploration_rate = exploration_rate
        self.min_score_threshold = min_score_threshold

        # affinity[code_idx][expert_id] = {cumulative_reward, visit_count, ema_reward}
        self.affinity: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)

    def get_best_expert(
        self,
        code_idx: int,
        explore: bool = True,
    ) -> Tuple[Optional[str], float]:
        """
        Get the best expert for a code based on accumulated affinity.

        Args:
            code_idx: The VQ code index
            explore: Whether to apply exploration randomness

        Returns:
            expert_id: Best expert, or None if no affinity exists or exploring
            score: The affinity score (0 if None)
        """
        if code_idx not in self.affinity or not self.affinity[code_idx]:
            return None, 0.0

        # Compute scores for all experts with affinity to this code
        scores = {}
        for expert_id, aff in self.affinity[code_idx].items():
            # Score = EMA reward * log(visit_count + 1) for stickiness
            # The log term makes established mappings resistant to change
            stickiness = math.log(aff['visit_count'] + 1) * self.stickiness_scale
            scores[expert_id] = aff['ema_reward'] * (1 + stickiness)

        best_expert = max(scores, key=scores.get)
        best_score = scores[best_expert]

        # Exploration: occasionally try alternatives
        if explore and random.random() < self.exploration_rate:
            logger.debug(f"Exploring alternative to {best_expert} for code {code_idx}")
            return None, 0.0

        # If best score is below threshold, explore via embedding similarity
        if best_score < self.min_score_threshold:
            return None, best_score

        return best_expert, best_score

    def update(
        self,
        code_idx: int,
        expert_id: str,
        reward: float,
    ):
        """
        Update affinity after a game is played.

        Args:
            code_idx: The VQ code that was active
            expert_id: The expert that played
            reward: The game reward achieved
        """
        if expert_id not in self.affinity[code_idx]:
            self.affinity[code_idx][expert_id] = {
                'cumulative_reward': 0.0,
                'visit_count': 0,
                'ema_reward': reward,  # Initialize with first reward
            }

        aff = self.affinity[code_idx][expert_id]
        aff['cumulative_reward'] += reward
        aff['visit_count'] += 1
        aff['ema_reward'] = (1 - self.ema_alpha) * aff['ema_reward'] + self.ema_alpha * reward

        logger.debug(
            f"Updated affinity: code {code_idx} -> {expert_id}: "
            f"ema_reward={aff['ema_reward']:.2f}, visits={aff['visit_count']}"
        )

    def get_stats(self) -> Dict:
        """Get affinity statistics."""
        total_mappings = sum(len(experts) for experts in self.affinity.values())
        codes_with_affinity = len(self.affinity)

        # Find most established mappings
        top_mappings = []
        for code_idx, experts in self.affinity.items():
            for expert_id, aff in experts.items():
                score = aff['ema_reward'] * math.log(aff['visit_count'] + 1)
                top_mappings.append((code_idx, expert_id, aff['visit_count'], aff['ema_reward'], score))

        top_mappings.sort(key=lambda x: x[4], reverse=True)

        return {
            'total_mappings': total_mappings,
            'codes_with_affinity': codes_with_affinity,
            'top_mappings': top_mappings[:10],  # Top 10 by score
        }

    def to_dict(self) -> Dict:
        """Serialize for saving."""
        return {
            'ema_alpha': self.ema_alpha,
            'stickiness_scale': self.stickiness_scale,
            'exploration_rate': self.exploration_rate,
            'min_score_threshold': self.min_score_threshold,
            'affinity': {
                str(code): {
                    expert_id: dict(aff)
                    for expert_id, aff in experts.items()
                }
                for code, experts in self.affinity.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeExpertAffinity':
        """Deserialize from saved data."""
        obj = cls(
            ema_alpha=data.get('ema_alpha', 0.1),
            stickiness_scale=data.get('stickiness_scale', 1.0),
            exploration_rate=data.get('exploration_rate', 0.1),
            min_score_threshold=data.get('min_score_threshold', 0.0),
        )

        for code_str, experts in data.get('affinity', {}).items():
            code_idx = int(code_str)
            for expert_id, aff in experts.items():
                obj.affinity[code_idx][expert_id] = dict(aff)

        return obj


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
        unified_action_space=None,
        # Affinity parameters
        affinity_ema_alpha: float = 0.1,
        affinity_stickiness: float = 1.0,
        affinity_exploration: float = 0.1,
    ):
        self.device = device
        self.epsilon = epsilon
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.num_actions = num_actions
        self.obs_shape = obs_shape
        self.max_experts_in_memory = max_experts_in_memory
        self.unified_action_space = unified_action_space

        # Tiered storage for experts
        self.storage = TieredStore(storage_path)

        # Reward-weighted code→expert affinity (replaces hard code_to_expert mapping)
        self.code_affinity = CodeExpertAffinity(
            ema_alpha=affinity_ema_alpha,
            stickiness_scale=affinity_stickiness,
            exploration_rate=affinity_exploration,
        )

        # Embedding centroids for each expert (for similarity matching)
        # expert_id -> centroid embedding
        self.expert_centroids: Dict[str, torch.Tensor] = {}

        # Currently loaded experts
        self.active_expert: Optional[Expert] = None
        self.active_expert_id: Optional[str] = None
        self.prefetched_expert: Optional[Expert] = None
        self.prefetched_expert_id: Optional[str] = None

        # Track current selection for affinity update
        self.current_code_idx: Optional[int] = None

        # Statistics
        self.stats = {
            'total_experts_created': 0,
            'total_swaps': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'affinity_selections': 0,
            'embedding_selections': 0,
            'new_expert_creations': 0,
        }

        # Load existing registry if present
        self._load_registry()

    def _load_registry(self):
        """Load expert registry from storage."""
        registry = self.storage.load_registry()
        if registry:
            # Load affinity data
            affinity_data = registry.get('code_affinity', None)
            if affinity_data:
                self.code_affinity = CodeExpertAffinity.from_dict(affinity_data)
                logger.info(f"Loaded affinity with {self.code_affinity.get_stats()['total_mappings']} mappings")

            # Load centroids
            centroids_data = registry.get('expert_centroids', {})
            for expert_id, centroid_list in centroids_data.items():
                self.expert_centroids[expert_id] = torch.tensor(
                    centroid_list, device=self.device
                )
            logger.info(f"Loaded registry with {len(self.expert_centroids)} expert centroids")

    def _save_registry(self):
        """Save expert registry to storage."""
        registry = {
            'code_affinity': self.code_affinity.to_dict(),
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

        Uses reward-weighted affinity for code→expert mapping, falling back
        to embedding similarity when exploring or when no affinity exists.

        Args:
            embedding: (code_dim,) game embedding from meta-agent
            code_idx: optional discrete code index

        Returns:
            Expert to use for this game
        """
        # Store code_idx for later affinity update
        self.current_code_idx = code_idx

        expert_id = None

        # First, try affinity-based selection if code is available
        if code_idx is not None:
            expert_id, affinity_score = self.code_affinity.get_best_expert(code_idx)

            if expert_id is not None:
                logger.info(f"Affinity selected {expert_id} for code {code_idx} (score={affinity_score:.3f})")
                self.stats['affinity_selections'] += 1

        # If no affinity match, try embedding similarity
        if expert_id is None:
            expert_id, similarity = self.find_similar_expert(embedding)

            if expert_id is not None:
                logger.info(f"Embedding selected {expert_id} (sim={similarity:.3f})")
                self.stats['embedding_selections'] += 1

        # If we found an expert, load it
        if expert_id is not None:
            # Check if already loaded
            if expert_id == self.active_expert_id:
                self.stats['cache_hits'] += 1
                return self.active_expert

            if expert_id == self.prefetched_expert_id:
                self.stats['cache_hits'] += 1
                self._swap_prefetched_to_active()
                return self.active_expert

            # Load from storage
            self.stats['cache_misses'] += 1
            return self._load_expert(expert_id)

        # No expert found - create new one
        logger.info(f"No suitable expert found, creating new")
        expert = self.create_expert()
        self.stats['new_expert_creations'] += 1

        # Register embedding as centroid for new expert
        self.expert_centroids[expert.expert_id] = embedding.detach().clone()

        # Set as active
        self._save_current_expert()
        self.active_expert = expert
        self.active_expert_id = expert.expert_id

        self._save_registry()

        return expert

    def update_affinity(self, reward: float, code_idx: Optional[int] = None):
        """
        Update code→expert affinity based on game reward.

        Call this after each game with the reward achieved.

        Args:
            reward: The reward achieved in the game
            code_idx: The code that was used (uses stored current_code_idx if None)
        """
        if code_idx is None:
            code_idx = self.current_code_idx

        if code_idx is None or self.active_expert_id is None:
            return

        self.code_affinity.update(code_idx, self.active_expert_id, reward)
        logger.debug(f"Updated affinity: code {code_idx} -> {self.active_expert_id}, reward={reward:.1f}")

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

    def get_active_expert(self) -> Optional[Expert]:
        """Get currently active expert."""
        return self.active_expert

    def get_stats(self) -> dict:
        """Get manager statistics."""
        affinity_stats = self.code_affinity.get_stats()
        return {
            **self.stats,
            'num_experts_stored': len(self.expert_centroids),
            'active_expert': self.active_expert_id,
            'prefetched_expert': self.prefetched_expert_id,
            'affinity_mappings': affinity_stats['total_mappings'],
            'codes_with_affinity': affinity_stats['codes_with_affinity'],
        }

    def get_affinity_stats(self) -> dict:
        """Get detailed affinity statistics."""
        return self.code_affinity.get_stats()

    def save_all(self):
        """Save all state to storage."""
        self._save_current_expert()
        self._save_registry()
        logger.info("Saved all expert manager state")

    def list_experts(self) -> List[str]:
        """List all stored expert IDs."""
        return list(self.expert_centroids.keys())

    def get_expert_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all experts (loads metadata from storage).

        Returns:
            Dict mapping expert_id to stats dict with:
            - total_frames: training frames
            - total_episodes: episodes trained
            - games_trained: list of game names
            - best_episode_reward: highest reward achieved
            - max_affinity_score: highest affinity score across all codes
            - total_affinity_visits: total visits across all code mappings
        """
        expert_stats = {}

        for expert_id in self.list_experts():
            stats = {
                'total_frames': 0,
                'total_episodes': 0,
                'games_trained': [],
                'best_episode_reward': float('-inf'),
                'max_affinity_score': 0.0,
                'total_affinity_visits': 0,
            }

            # Load metadata from storage
            data = self.storage.load_expert(expert_id)
            if data and 'metadata' in data:
                meta = data['metadata']
                stats['total_frames'] = meta.get('total_frames', 0)
                stats['total_episodes'] = meta.get('total_episodes', 0)
                stats['games_trained'] = meta.get('games_trained', [])
                stats['best_episode_reward'] = meta.get('best_episode_reward', float('-inf'))

            # Compute affinity stats
            for code_idx, experts in self.code_affinity.affinity.items():
                if expert_id in experts:
                    aff = experts[expert_id]
                    stickiness = math.log(aff['visit_count'] + 1) * self.code_affinity.stickiness_scale
                    score = aff['ema_reward'] * (1 + stickiness)
                    stats['max_affinity_score'] = max(stats['max_affinity_score'], score)
                    stats['total_affinity_visits'] += aff['visit_count']

            expert_stats[expert_id] = stats

        return expert_stats

    def prune_experts(
        self,
        min_frames: int = 50000,
        min_affinity_score: float = 0.0,
        max_experts: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict:
        """
        Prune undertrained experts to reduce fragmentation.

        Experts are removed if they have:
        - Fewer than min_frames training frames, AND
        - Max affinity score below min_affinity_score

        Optionally, can also enforce a max_experts limit by removing
        the lowest-scoring experts until the limit is met.

        Args:
            min_frames: Minimum training frames to keep expert
            min_affinity_score: Minimum affinity score to keep expert
            max_experts: Optional maximum number of experts to keep
            dry_run: If True, just report what would be pruned

        Returns:
            Dict with pruning results:
            - pruned: list of pruned expert IDs
            - kept: list of kept expert IDs
            - protected: list of protected expert IDs (active/prefetched)
        """
        expert_stats = self.get_expert_stats()

        # Compute composite score for each expert
        # Score = frames * (1 + max_affinity_score) to favor both training and good performance
        scored_experts = []
        for expert_id, stats in expert_stats.items():
            composite_score = stats['total_frames'] * (1 + max(0, stats['max_affinity_score']))
            scored_experts.append((expert_id, stats, composite_score))

        # Sort by score descending
        scored_experts.sort(key=lambda x: x[2], reverse=True)

        # Determine which experts to prune
        protected = set()
        if self.active_expert_id:
            protected.add(self.active_expert_id)
        if self.prefetched_expert_id:
            protected.add(self.prefetched_expert_id)

        to_prune = []
        to_keep = []

        for expert_id, stats, score in scored_experts:
            if expert_id in protected:
                to_keep.append(expert_id)
                continue

            # Check if expert meets minimum requirements
            meets_frames = stats['total_frames'] >= min_frames
            meets_affinity = stats['max_affinity_score'] >= min_affinity_score

            # Keep if meets either threshold (frames OR affinity)
            if meets_frames or meets_affinity:
                to_keep.append(expert_id)
            else:
                to_prune.append(expert_id)

        # If max_experts is set, also prune lowest-scoring non-protected experts
        if max_experts is not None:
            current_count = len(to_keep)
            if current_count > max_experts:
                # Already keeping too many, need to prune more
                # Sort kept experts by score and move lowest to prune list
                kept_with_scores = [
                    (eid, next(s for e, s, _ in scored_experts if e == eid))
                    for eid in to_keep if eid not in protected
                ]
                kept_with_scores.sort(
                    key=lambda x: x[1]['total_frames'] * (1 + max(0, x[1]['max_affinity_score']))
                )

                excess = current_count - max_experts
                for eid, _ in kept_with_scores[:excess]:
                    to_keep.remove(eid)
                    to_prune.append(eid)

        if not dry_run and to_prune:
            for expert_id in to_prune:
                self._delete_expert(expert_id)
            logger.info(f"Pruned {len(to_prune)} experts: {to_prune}")

        return {
            'pruned': to_prune,
            'kept': to_keep,
            'protected': list(protected),
            'total_before': len(expert_stats),
            'total_after': len(to_keep),
        }

    def _delete_expert(self, expert_id: str):
        """
        Delete an expert and clean up all references.

        Redistributes affinity mappings to remaining similar experts.
        """
        # Remove from centroids
        if expert_id in self.expert_centroids:
            deleted_centroid = self.expert_centroids.pop(expert_id)
        else:
            deleted_centroid = None

        # Remove from affinity mappings
        # For each code that mapped to this expert, remove the mapping
        # The next selection will fall back to embedding similarity
        codes_to_clean = []
        for code_idx, experts in self.code_affinity.affinity.items():
            if expert_id in experts:
                codes_to_clean.append((code_idx, experts[expert_id]))

        for code_idx, old_aff in codes_to_clean:
            del self.code_affinity.affinity[code_idx][expert_id]
            logger.debug(f"Removed affinity mapping: code {code_idx} -> {expert_id}")

            # Optionally: redistribute affinity to most similar remaining expert
            if deleted_centroid is not None and self.expert_centroids:
                similar_expert, similarity = self.find_similar_expert(deleted_centroid)
                if similar_expert is not None and similarity > 0.5:
                    # Transfer some affinity to similar expert
                    if similar_expert not in self.code_affinity.affinity[code_idx]:
                        self.code_affinity.affinity[code_idx][similar_expert] = {
                            'cumulative_reward': 0.0,
                            'visit_count': 0,
                            'ema_reward': old_aff['ema_reward'] * 0.5,  # Discount transferred affinity
                        }
                    logger.debug(f"Transferred affinity from {expert_id} to {similar_expert} for code {code_idx}")

        # Delete from storage
        self.storage.delete_expert(expert_id)

        # Save updated registry
        self._save_registry()

        logger.info(f"Deleted expert {expert_id}")

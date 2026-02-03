"""
UnifiedActionSpace: Semantic action mapping across Atari games.

Builds the union of action meanings across all training games.
Each semantic action (e.g. 'FIRE') gets one unified index.
Per-game masks and mapping tables enable correct action selection.
"""

import torch
import gymnasium as gym
from typing import List, Dict
import logging

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Full ALE action vocabulary in canonical order
ALE_ACTION_MEANINGS = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
    'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
    'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE',
]


class UnifiedActionSpace:
    """
    Unified action space across multiple Atari games.

    Takes the union of action meanings from all training games,
    merges semantically identical actions to the same index,
    and provides per-game masks and mapping tables.
    """

    def __init__(self, game_names: List[str]):
        self.game_names = game_names

        # Query each game for its action meanings
        self.game_action_meanings: Dict[str, List[str]] = {}
        for game in game_names:
            self.game_action_meanings[game] = self._get_action_meanings(game)

        # Build union: keep canonical ALE order, filtered to actions seen in any game
        seen = set()
        for meanings in self.game_action_meanings.values():
            seen.update(meanings)

        self.unified_actions: List[str] = [
            a for a in ALE_ACTION_MEANINGS if a in seen
        ]
        self.num_actions = len(self.unified_actions)

        # Name -> unified index lookup
        self.action_to_unified: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.unified_actions)
        }

        # Per-game artifacts
        self.action_masks: Dict[str, torch.Tensor] = {}
        self.unified_to_local: Dict[str, torch.Tensor] = {}
        self.local_to_unified: Dict[str, torch.Tensor] = {}

        for game in game_names:
            meanings = self.game_action_meanings[game]

            mask = torch.zeros(self.num_actions, dtype=torch.bool)
            u2l = torch.full((self.num_actions,), -1, dtype=torch.long)
            l2u = torch.zeros(len(meanings), dtype=torch.long)

            for local_idx, name in enumerate(meanings):
                unified_idx = self.action_to_unified[name]
                mask[unified_idx] = True
                u2l[unified_idx] = local_idx
                l2u[local_idx] = unified_idx

            self.action_masks[game] = mask
            self.unified_to_local[game] = u2l
            self.local_to_unified[game] = l2u

        logger.info(
            f"Unified action space: {self.num_actions} actions "
            f"from {len(game_names)} games: {self.unified_actions}"
        )

        # Print summary
        for game in game_names:
            valid = [
                self.unified_actions[i]
                for i in range(self.num_actions)
                if self.action_masks[game][i]
            ]
            logger.info(f"  {game}: {len(valid)} valid actions: {valid}")

    def _get_action_meanings(self, game_name: str) -> List[str]:
        """Query ALE for a game's action meanings."""
        for game_id in [f"ALE/{game_name}-v5", f"{game_name}NoFrameskip-v4"]:
            try:
                env = gym.make(game_id)
                meanings = env.unwrapped.get_action_meanings()
                env.close()
                return meanings
            except Exception:
                continue
        raise RuntimeError(f"Could not get action meanings for {game_name}")

    def get_mask(self, game_name: str, device: torch.device) -> torch.Tensor:
        """Get bool action mask for a game on specified device."""
        return self.action_masks[game_name].to(device)

    def unified_to_local_action(self, unified_action: int, game_name: str) -> int:
        """Convert unified action index to local game index."""
        local = self.unified_to_local[game_name][unified_action].item()
        if local < 0:
            raise ValueError(
                f"Action {unified_action} ({self.unified_actions[unified_action]}) "
                f"is not valid for {game_name}"
            )
        return local

    def apply_mask(self, logits: torch.Tensor, game_name: str) -> torch.Tensor:
        """Set invalid action logits to -inf."""
        mask = self.action_masks[game_name].to(logits.device)
        mask = mask.unsqueeze(0).expand_as(logits)
        return logits.masked_fill(~mask, float('-inf'))

    def to_device(self, device: torch.device):
        """Pre-move all tensors to device for performance."""
        for game in self.game_names:
            self.action_masks[game] = self.action_masks[game].to(device)
            self.unified_to_local[game] = self.unified_to_local[game].to(device)
            self.local_to_unified[game] = self.local_to_unified[game].to(device)

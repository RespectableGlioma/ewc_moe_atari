"""
GameCurriculum: Manages game sequences for meta-agent training.

Supports different curriculum types:
- Random: Uniform sampling
- Markov: First-order transition probabilities
- Periodic: Deterministic cycles
- Contextual: Higher-order dependencies
"""

import numpy as np
from enum import Enum
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CurriculumType(Enum):
    RANDOM = "random"
    MARKOV = "markov"
    PERIODIC = "periodic"
    CONTEXTUAL = "contextual"


class GameCurriculum:
    """
    Manages game sequences for training.

    The curriculum type affects what the meta-agent can learn:
    - Random: Just detection, no prefetch benefit
    - Markov: First-order transition probabilities
    - Periodic: Deterministic cycles (A→B→C→A...)
    - Contextual: Pattern depends on meta-state
    """

    def __init__(
        self,
        games: List[str],
        curriculum_type: CurriculumType = CurriculumType.RANDOM,
        seed: Optional[int] = None,
        transition_matrix: Optional[np.ndarray] = None,
        periodic_sequence: Optional[List[int]] = None,
    ):
        self.games = games
        self.n_games = len(games)
        self.curriculum_type = curriculum_type
        self.rng = np.random.default_rng(seed)

        self.current_idx = 0
        self.history: List[int] = []

        # For Markov curriculum
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Default: uniform transitions
            self.transition_matrix = np.ones((self.n_games, self.n_games)) / self.n_games

        # For periodic curriculum
        if periodic_sequence is not None:
            self.periodic_sequence = periodic_sequence
        else:
            # Default: cycle through all games
            self.periodic_sequence = list(range(self.n_games))

        self.periodic_pos = 0

        # For contextual curriculum
        self.context_state = 0

        logger.info(f"Created curriculum: {curriculum_type.value} over {len(games)} games")

    def reset(self):
        """Reset curriculum state."""
        self.current_idx = 0
        self.history = []
        self.periodic_pos = 0
        self.context_state = 0

    def get_current_game(self) -> str:
        """Get current game name."""
        return self.games[self.current_idx]

    def get_current_idx(self) -> int:
        """Get current game index."""
        return self.current_idx

    def next_game(self) -> str:
        """
        Transition to next game based on curriculum type.

        Returns:
            Name of the next game
        """
        self.history.append(self.current_idx)

        if self.curriculum_type == CurriculumType.RANDOM:
            self.current_idx = self.rng.integers(0, self.n_games)

        elif self.curriculum_type == CurriculumType.MARKOV:
            probs = self.transition_matrix[self.current_idx]
            self.current_idx = self.rng.choice(self.n_games, p=probs)

        elif self.curriculum_type == CurriculumType.PERIODIC:
            self.periodic_pos = (self.periodic_pos + 1) % len(self.periodic_sequence)
            self.current_idx = self.periodic_sequence[self.periodic_pos]

        elif self.curriculum_type == CurriculumType.CONTEXTUAL:
            self.current_idx = self._contextual_next()

        return self.games[self.current_idx]

    def _contextual_next(self) -> int:
        """
        Contextual curriculum with higher-order dependencies.

        Example: alternating between difficulty levels based on
        performance or time spent in current context.
        """
        # Simple implementation: 2-state context
        # State 0: choose from first half of games
        # State 1: choose from second half
        # Transition between states with some probability

        if self.rng.random() < 0.3:
            self.context_state = 1 - self.context_state

        half = self.n_games // 2
        if self.context_state == 0:
            return self.rng.integers(0, half)
        else:
            return self.rng.integers(half, self.n_games)

    def get_transition_distribution(self) -> np.ndarray:
        """
        Get transition probability distribution from current game.

        Returns:
            (n_games,) probability distribution
        """
        if self.curriculum_type == CurriculumType.RANDOM:
            return np.ones(self.n_games) / self.n_games

        elif self.curriculum_type == CurriculumType.MARKOV:
            return self.transition_matrix[self.current_idx]

        elif self.curriculum_type == CurriculumType.PERIODIC:
            # Deterministic: 1.0 for next in sequence
            next_pos = (self.periodic_pos + 1) % len(self.periodic_sequence)
            next_idx = self.periodic_sequence[next_pos]
            dist = np.zeros(self.n_games)
            dist[next_idx] = 1.0
            return dist

        elif self.curriculum_type == CurriculumType.CONTEXTUAL:
            # Approximate distribution
            dist = np.zeros(self.n_games)
            half = self.n_games // 2
            if self.context_state == 0:
                dist[:half] = 1.0 / half
            else:
                dist[half:] = 1.0 / (self.n_games - half)
            return dist

        return np.ones(self.n_games) / self.n_games

    def generate_sequence(self, length: int) -> List[str]:
        """
        Generate a sequence of games.

        Args:
            length: Number of games in sequence

        Returns:
            List of game names
        """
        sequence = []
        for _ in range(length):
            sequence.append(self.get_current_game())
            self.next_game()
        return sequence

    def get_stats(self) -> Dict:
        """Get curriculum statistics."""
        game_counts = np.zeros(self.n_games)
        for idx in self.history:
            game_counts[idx] += 1

        return {
            'total_games': len(self.history),
            'game_counts': {
                self.games[i]: int(game_counts[i])
                for i in range(self.n_games)
            },
            'curriculum_type': self.curriculum_type.value,
        }


def create_markov_curriculum(
    games: List[str],
    sparsity: float = 0.3,
    seed: Optional[int] = None,
) -> GameCurriculum:
    """
    Create a Markov curriculum with sparse transition matrix.

    Games tend to transition to a subset of "similar" games.
    """
    rng = np.random.default_rng(seed)
    n_games = len(games)

    # Create sparse transition matrix
    matrix = np.zeros((n_games, n_games))

    for i in range(n_games):
        # Each game transitions to ~sparsity fraction of games
        n_transitions = max(1, int(n_games * sparsity))
        targets = rng.choice(n_games, size=n_transitions, replace=False)
        probs = rng.dirichlet(np.ones(n_transitions))
        matrix[i, targets] = probs

    return GameCurriculum(
        games=games,
        curriculum_type=CurriculumType.MARKOV,
        transition_matrix=matrix,
        seed=seed,
    )


def create_periodic_curriculum(
    games: List[str],
    pattern: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> GameCurriculum:
    """
    Create a periodic curriculum that cycles through games.

    Args:
        games: List of game names
        pattern: Optional custom pattern (indices into games list)
        seed: Random seed
    """
    if pattern is None:
        # Default: random permutation then repeat
        rng = np.random.default_rng(seed)
        pattern = list(rng.permutation(len(games)))

    return GameCurriculum(
        games=games,
        curriculum_type=CurriculumType.PERIODIC,
        periodic_sequence=pattern,
        seed=seed,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_obs: np.ndarray
    salience: float


class _GameEpisodes:
    def __init__(self):
        self.episodes: List[List[Transition]] = []
        self._current: List[Transition] = []

    def add(self, tr: Transition) -> None:
        self._current.append(tr)

    def end_episode(self) -> None:
        if self._current:
            self.episodes.append(self._current)
        self._current = []

    def flush(self) -> None:
        self.episodes = []
        self._current = []

    def num_transitions(self) -> int:
        return sum(len(ep) for ep in self.episodes) + len(self._current)

    def finalize(self) -> None:
        # Make sure current incomplete episode is committed
        if self._current:
            self.episodes.append(self._current)
            self._current = []

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        *,
        prioritize: bool = True,
        alpha: float = 0.6,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Samples (batch, seq_len) sequences.

        This is a simple (rebuild-each-sample) prioritized sampler intended for day-sized buffers.
        Priority for a sequence is max salience in the window.
        """
        self.finalize()
        rng = rng or np.random.default_rng()

        candidates: List[Tuple[int, int]] = []  # (episode_idx, start)
        weights: List[float] = []

        for epi, ep in enumerate(self.episodes):
            if len(ep) < seq_len:
                continue
            for start in range(0, len(ep) - seq_len + 1):
                candidates.append((epi, start))
                if prioritize:
                    w = max(float(t.salience) for t in ep[start : start + seq_len])
                    weights.append((w + 1e-6) ** alpha)

        if not candidates:
            raise RuntimeError(
                f"Not enough data to sample sequences (need episodes with len >= {seq_len})."
            )

        if prioritize:
            probs = np.asarray(weights, dtype=np.float64)
            probs = probs / probs.sum()
            idxs = rng.choice(len(candidates), size=batch_size, replace=True, p=probs)
        else:
            idxs = rng.integers(0, len(candidates), size=batch_size)

        # Allocate
        first_tr = self.episodes[candidates[int(idxs[0])][0]][candidates[int(idxs[0])][1]]
        obs_shape = first_tr.obs.shape

        obs = np.zeros((batch_size, seq_len) + obs_shape, dtype=first_tr.obs.dtype)
        next_obs = np.zeros((batch_size, seq_len) + obs_shape, dtype=first_tr.next_obs.dtype)
        actions = np.zeros((batch_size, seq_len), dtype=np.int64)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len), dtype=np.float32)
        salience = np.zeros((batch_size, seq_len), dtype=np.float32)

        for bi, ci in enumerate(idxs):
            epi, start = candidates[int(ci)]
            window = self.episodes[epi][start : start + seq_len]
            for t, tr in enumerate(window):
                obs[bi, t] = tr.obs
                next_obs[bi, t] = tr.next_obs
                actions[bi, t] = tr.action
                rewards[bi, t] = tr.reward
                done = tr.terminated or tr.truncated
                dones[bi, t] = 1.0 if done else 0.0
                salience[bi, t] = tr.salience

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_obs": next_obs,
            "salience": salience,
        }


class SleepBuffer:
    """Stores only the games encountered during the day, then can be flushed.

    This is the 'sleep buffer'. It is intentionally short-lived.
    """

    def __init__(self):
        self.games: Dict[str, _GameEpisodes] = {}

    def _get(self, game_id: str) -> _GameEpisodes:
        if game_id not in self.games:
            self.games[game_id] = _GameEpisodes()
        return self.games[game_id]

    def add(self, game_id: str, tr: Transition) -> None:
        self._get(game_id).add(tr)

    def end_episode(self, game_id: str) -> None:
        self._get(game_id).end_episode()

    def list_games(self) -> List[str]:
        return list(self.games.keys())

    def num_transitions(self, game_id: Optional[str] = None) -> int:
        if game_id is None:
            return sum(g.num_transitions() for g in self.games.values())
        return self._get(game_id).num_transitions()

    def sample_sequences(self, game_id: str, batch_size: int, seq_len: int, **kwargs) -> Dict[str, np.ndarray]:
        return self._get(game_id).sample_sequences(batch_size=batch_size, seq_len=seq_len, **kwargs)

    def flush(self) -> None:
        for g in self.games.values():
            g.flush()
        self.games = {}

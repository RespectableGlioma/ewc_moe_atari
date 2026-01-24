from __future__ import annotations

"""Atari (ALE) helpers.

This module provides a minimal batching interface so the rest of the prototype
can treat "real" observations similarly to the synthetic stream.

Key design choices:

* We keep a *small* set of actual Atari games (e.g. Pong, Breakout, ...).
* We allow a *large* number of logical env IDs (e.g. 512) by mapping
  ``env_id -> game_idx = env_id % len(games)``. This is useful for stressing
  the out-of-core expert cache at scale without spinning up hundreds of unique
  simulator instances.

Two backends are supported:

* ``backend='ale_vec'`` (default): uses ALE's C++ AtariVectorEnv via
  ``gymnasium.make_vec``.
* ``backend='wrappers'``: uses Gymnasium's Python wrappers
  ``AtariPreprocessing`` + ``FrameStackObservation`` wrapped around
  ``gymnasium.make(..., frameskip=1)`` and vectorized with SyncVectorEnv.

The 'wrappers' backend is included because Gymnasium v1.x renamed the classic
``FrameStack`` wrapper to ``FrameStackObservation``.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _require_gymnasium_and_ale() -> "tuple[object, object]":
    try:
        import gymnasium as gym  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "gymnasium is required for Atari integration. Install with: pip install 'gymnasium[atari]'"
        ) from e

    try:
        import ale_py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ale-py is required for Atari integration. Install with: pip install 'gymnasium[atari]'"
        ) from e

    # Gymnasium >=1.0 removed the implicit plugin import that used to register
    # Atari environments. This call is effectively a no-op but makes IDEs and
    # linters keep the import and ensures users do the right thing.
    try:
        gym.register_envs(ale_py)  # type: ignore[attr-defined]
    except Exception:
        pass

    return gym, ale_py


def _normalize_game_id(game: str) -> str:
    """Normalize a game spec into a Gymnasium env_id string.

    Accepts:
      - "ALE/Pong-v5" (returned unchanged)
      - "ale_py:ALE/Pong-v5" (returned unchanged)
      - "Pong" -> "ALE/Pong-v5"
      - "Pong-v5" -> "ALE/Pong-v5"
    """
    game = game.strip()
    if ":" in game:
        return game
    if game.startswith("ALE/"):
        return game
    if game.startswith("ale_py:"):
        return game
    if game.startswith("PongNoFrameskip"):
        # Old gym-style IDs are not guaranteed in Gymnasium; keep as-is if user asked.
        return game
    if "-v" in game:
        return f"ALE/{game}"
    return f"ALE/{game}-v5"


@dataclass
class AtariEnvConfig:
    backend: str = "ale_vec"  # 'ale_vec' | 'wrappers'
    num_envs: int = 8
    frameskip: int = 4
    stack_size: int = 4
    noop_max: int = 30
    img_height: int = 84
    img_width: int = 84
    grayscale: bool = True
    repeat_action_probability: float = 0.0
    use_fire_reset: bool = False
    reward_clipping: bool = False
    episodic_life: bool = False


class AtariBatchSource:
    """A minimal batch sampler over a small set of Atari games.

    Methods:
      - sample(env_id, n): returns an array of shape (n, stack, H, W) (grayscale)
        or a best-effort equivalent depending on backend / settings.
    """

    def __init__(
        self,
        games: List[str],
        *,
        cfg: AtariEnvConfig,
        seed: int = 0,
    ):
        if not games:
            raise ValueError("games must be non-empty")
        if cfg.backend not in ("ale_vec", "wrappers"):
            raise ValueError("cfg.backend must be one of: ale_vec, wrappers")
        if cfg.num_envs <= 0:
            raise ValueError("cfg.num_envs must be > 0")

        gym, _ale_py = _require_gymnasium_and_ale()
        self._gym = gym

        self.games_raw = list(games)
        self.env_ids = [_normalize_game_id(g) for g in games]
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # One vector env per game.
        self.envs = []
        for gi, env_id in enumerate(self.env_ids):
            env = self._make_env(env_id, game_index=gi)
            # Prime with a reset to have a valid internal state.
            try:
                env.reset(seed=self.seed + 17 * gi)
            except TypeError:
                env.reset()
            self.envs.append(env)

        # Cached action metadata (best-effort).
        self._n_actions: Dict[int, Optional[int]] = {i: None for i in range(len(self.envs))}
        for i, env in enumerate(self.envs):
            n = None
            try:
                # Gymnasium VectorEnv has single_action_space in v1.x
                sa = getattr(env, "single_action_space", None)
                if sa is not None and getattr(sa, "n", None) is not None:
                    n = int(sa.n)
                else:
                    # Some vector envs expose an action_space that samples batched actions.
                    a0 = env.action_space.sample()
                    # If sampling returns a scalar, treat as Discrete.
                    if np.isscalar(a0):
                        n = int(getattr(env.action_space, "n", 0)) or None
            except Exception:
                n = None
            self._n_actions[i] = n

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    def _make_env(self, env_id: str, *, game_index: int):
        gym = self._gym
        cfg = self.cfg

        if cfg.backend == "ale_vec":
            # Prefer ALE's C++ vector environment via gym.make_vec (fast).
            # Based on ALE Vector Environment Guide.
            kwargs = dict(
                num_envs=int(cfg.num_envs),
                frameskip=int(cfg.frameskip),
                stack_num=int(cfg.stack_size),
                noop_max=int(cfg.noop_max),
                img_height=int(cfg.img_height),
                img_width=int(cfg.img_width),
                grayscale=bool(cfg.grayscale),
                reward_clipping=bool(cfg.reward_clipping),
                use_fire_reset=bool(cfg.use_fire_reset),
                repeat_action_probability=float(cfg.repeat_action_probability),
                episodic_life=bool(cfg.episodic_life),
            )

            try:
                return gym.make_vec(env_id, **kwargs)
            except TypeError:
                # Fallback for older gymnasium/ale-py versions that may not expose
                # all AtariVectorEnv kwargs through make_vec.
                return gym.make_vec(
                    env_id,
                    num_envs=int(cfg.num_envs),
                    use_fire_reset=bool(cfg.use_fire_reset),
                    reward_clipping=bool(cfg.reward_clipping),
                    repeat_action_probability=float(cfg.repeat_action_probability),
                )

        # cfg.backend == 'wrappers'
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        def _thunk(idx: int):
            # Important: disable frame skipping in base env (frameskip=1) because
            # AtariPreprocessing does its own frame skipping.
            base = gym.make(
                env_id,
                frameskip=1,
                repeat_action_probability=float(cfg.repeat_action_probability),
            )
            env = AtariPreprocessing(
                base,
                noop_max=int(cfg.noop_max),
                frame_skip=int(cfg.frameskip),
                screen_size=(int(cfg.img_width), int(cfg.img_height)),
                terminal_on_life_loss=bool(cfg.episodic_life),
                grayscale_obs=bool(cfg.grayscale),
                grayscale_newaxis=False,
                scale_obs=False,
            )
            env = FrameStackObservation(env, stack_size=int(cfg.stack_size), padding_type="zero")
            try:
                env.reset(seed=self.seed + 1000 * game_index + idx)
            except TypeError:
                env.reset()
            return env

        fns = [lambda i=i: _thunk(i) for i in range(int(cfg.num_envs))]
        return gym.vector.SyncVectorEnv(fns)

    def _sample_actions(self, game_index: int, num_envs: int):
        n = self._n_actions.get(game_index, None)
        if n is None or n <= 0:
            # Fall back to gym's sampler.
            return self.envs[game_index].action_space.sample()
        return self.rng.integers(0, n, size=(num_envs,), dtype=np.int64)

    def env_id_to_game_index(self, env_id: int) -> int:
        return int(env_id) % len(self.envs)

    def sample(self, env_id: int, n: int) -> np.ndarray:
        """Return `n` stacked observations (uint8) for the given logical env_id."""
        if n <= 0:
            raise ValueError("n must be > 0")

        gi = self.env_id_to_game_index(env_id)
        env = self.envs[gi]
        num_envs = int(getattr(env, "num_envs", self.cfg.num_envs))

        obs_chunks: List[np.ndarray] = []
        got = 0
        while got < n:
            actions = self._sample_actions(gi, num_envs)
            obs, _rew, _term, _trunc, _info = env.step(actions)
            obs_np = np.asarray(obs)
            obs_chunks.append(obs_np)
            got += int(obs_np.shape[0])

        out = np.concatenate(obs_chunks, axis=0)[:n]
        # Most Atari observations are uint8; keep as-is.
        return out

from __future__ import annotations

from typing import Optional

import gymnasium as gym

from gymnasium.wrappers import AtariPreprocessing

# Gymnasium v1.0+ uses FrameStackObservation (FrameStack was renamed).
try:
    from gymnasium.wrappers import FrameStackObservation
except Exception:  # pragma: no cover
    # Fallback for older gymnasium, but the notebook installs v1+
    from gymnasium.wrappers import FrameStack as FrameStackObservation  # type: ignore

from .wrappers import ClipReward, AddGameIdToInfo
from utils.seed import maybe_seed_env


def make_atari_env(
    env_id: str,
    *,
    seed: Optional[int] = None,
    frame_stack: int = 4,
    clip_rewards: bool = True,
    full_action_space: bool = True,
    disable_sticky_actions: bool = True,
):
    """Creates a Gymnasium Atari env with DQN-style preprocessing.

    Design goals:
    - unified action space via `full_action_space=True` => Discrete(18)
    - modern Gymnasium wrapper `FrameStackObservation`
    - reward clipping (optional)

    Notes:
    - AtariPreprocessing performs frame skipping, grayscaling, resizing, etc.
    - FrameStackObservation stacks the last N observations.
    """

    # Optional: register ALE envs explicitly for IDEs / robustness.
    try:
        import ale_py  # noqa: F401

        gym.register_envs(ale_py)
    except Exception:
        pass

    # Some Atari env kwargs (repeat_action_probability controls "sticky actions").
    #
    # IMPORTANT: AtariPreprocessing applies its own frame skipping. The wrapped ALE env
    # must have internal frame skipping disabled (frameskip=1), otherwise Gymnasium
    # raises a ValueError to avoid double-skipping.
    make_kwargs = {
        "frameskip": 1,
        "full_action_space": bool(full_action_space),
    }
    if disable_sticky_actions:
        make_kwargs["repeat_action_probability"] = 0.0

    env = gym.make(env_id, **make_kwargs)
    maybe_seed_env(env, seed)

    # Common DQN-ish preprocessing.
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    env = FrameStackObservation(env, stack_size=int(frame_stack), padding_type="reset")

    if clip_rewards:
        env = ClipReward(env, -1.0, 1.0)

    env = AddGameIdToInfo(env, env_id)

    # Sanity check: unified action space when requested.
    if full_action_space:
        assert (
            env.action_space.n == 18
        ), f"Expected Discrete(18) with full_action_space=True, got {env.action_space}"  # noqa: E501

    return env

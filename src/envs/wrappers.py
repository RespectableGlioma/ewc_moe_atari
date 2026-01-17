from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class ClipReward(gym.RewardWrapper):
    """Clips rewards into [min_reward, max_reward]."""

    def __init__(self, env: gym.Env, min_reward: float = -1.0, max_reward: float = 1.0):
        super().__init__(env)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.min_reward, self.max_reward))


class AddGameIdToInfo(gym.Wrapper):
    """Adds a 'game_id' field into the info dict for every step/reset.

    This is convenient when you want to log or feed context without touching the observation.
    """

    def __init__(self, env: gym.Env, game_id: str):
        super().__init__(env)
        self.game_id = game_id

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["game_id"] = self.game_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["game_id"] = self.game_id
        return obs, reward, terminated, truncated, info

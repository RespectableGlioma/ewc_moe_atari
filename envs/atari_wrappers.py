"""
Atari environment wrappers.

Standard preprocessing for Atari games following best practices.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, List
from collections import deque


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every skip-th frame, take max over last 2 frames.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take FIRE action on reset for environments that require it.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 grayscale.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        import cv2
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs


class FrameStack(gym.Wrapper):
    """
    Stack k last frames. Returns lazy array with shape (k, H, W).
    """

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(k,) + shp,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.frames)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip reward to {-1, 0, +1} based on sign.
    """

    def reward(self, reward):
        return np.sign(reward)


class AtariWrapper(gym.Wrapper):
    """
    Wrapper that applies all standard Atari preprocessing.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        frame_stack: int = 4,
        clip_reward: bool = True,
        terminal_on_life_loss: bool = True,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)
        env = FrameStack(env, k=frame_stack)
        super().__init__(env)


def make_atari_env(
    game_name: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    """
    Create a wrapped Atari environment.

    Args:
        game_name: Name of the Atari game (e.g., "Breakout", "Pong")
        seed: Random seed
        render_mode: Gymnasium render mode
        **kwargs: Additional arguments to AtariWrapper

    Returns:
        Wrapped Atari environment
    """
    # Ensure proper game ID format
    if not game_name.endswith("NoFrameskip-v4"):
        game_id = f"{game_name}NoFrameskip-v4"
    else:
        game_id = game_name

    env = gym.make(game_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)

    env = AtariWrapper(env, **kwargs)
    return env


# List of common Atari games for curriculum
ATARI_GAMES = [
    "Breakout",
    "Pong",
    "SpaceInvaders",
    "Seaquest",
    "BeamRider",
    "Enduro",
    "Qbert",
    "Asterix",
    "MsPacman",
    "Freeway",
    "Boxing",
    "Bowling",
    "Frostbite",
    "Gravitar",
    "Kangaroo",
]


def get_num_actions(game_name: str) -> int:
    """Get number of actions for a specific game."""
    env = gym.make(f"{game_name}NoFrameskip-v4")
    num_actions = env.action_space.n
    env.close()
    return num_actions

"""
Atari environment wrappers and utilities.

Standard preprocessing:
1. NoopReset: Random number of no-ops at start
2. MaxAndSkip: Take max over last 2 frames, skip 4 frames
3. FireReset: Press FIRE on reset if needed
4. Grayscale + Resize: Convert to 84x84 grayscale
5. FrameStack: Stack 4 frames for temporal info
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is action 0.
    """
    
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return max pooled frame over last 2 observations.
    Skip 4 frames per action (frame skipping).
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        total_reward = 0.0
        terminated = truncated = False
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take FIRE action on reset for games that require it.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT (or other)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class GrayscaleResizeEnv(gym.ObservationWrapper):
    """
    Convert to grayscale and resize to 84x84.
    """
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width),
            dtype=np.uint8
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        
        # Convert to grayscale
        if len(obs.shape) == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs


class FrameStackEnv(gym.Wrapper):
    """
    Stack last k frames.
    """
    
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(k, shp[0], shp[1]),
            dtype=np.uint8
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, 1}.
    """
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)


def make_atari_env(
    env_name: str,
    frame_stack: int = 4,
    clip_rewards: bool = True,
    episodic_life: bool = True,
) -> gym.Env:
    """
    Create a wrapped Atari environment with standard preprocessing.
    
    Args:
        env_name: Name of Atari game (e.g., "BreakoutNoFrameskip-v4")
        frame_stack: Number of frames to stack
        clip_rewards: Whether to clip rewards
        episodic_life: Whether to use episodic life wrapper
        
    Returns:
        Wrapped environment
    """
    env = gym.make(env_name)
    
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = GrayscaleResizeEnv(env, width=84, height=84)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    env = FrameStackEnv(env, k=frame_stack)
    
    return env


# Standard Atari-57 games (subset for prototyping)
ATARI_GAMES = [
    "Breakout",
    "Pong",
    "SpaceInvaders",
    "Seaquest",
    "BeamRider",
    "Qbert",
    "Asteroids",
    "Centipede",
    "MsPacman",
    "Frostbite",
]

# Full Atari-57 list
ATARI_57 = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids",
    "Atlantis", "BankHeist", "BattleZone", "BeamRider", "Berzerk",
    "Bowling", "Boxing", "Breakout", "Centipede", "ChopperCommand",
    "CrazyClimber", "Defender", "DemonAttack", "DoubleDunk", "Enduro",
    "FishingDerby", "Freeway", "Frostbite", "Gopher", "Gravitar",
    "Hero", "IceHockey", "Jamesbond", "Kangaroo", "Krull",
    "KungFuMaster", "MontezumaRevenge", "MsPacman", "NameThisGame", "Phoenix",
    "Pitfall", "Pong", "PrivateEye", "Qbert", "Riverraid",
    "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris",
    "SpaceInvaders", "StarGunner", "Surround", "Tennis", "TimePilot",
    "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor",
    "YarsRevenge", "Zaxxon",
]


def get_atari_env_id(game_name: str) -> str:
    """Convert game name to Gymnasium env ID."""
    return f"ALE/{game_name}-v5"


class DummyAtariEnv(gym.Env):
    """
    Dummy Atari-like environment for testing without ALE.
    
    Simulates the observation space and basic dynamics.
    """
    
    def __init__(
        self,
        game_id: int = 0,
        obs_shape: Tuple[int, int, int] = (4, 84, 84),
        num_actions: int = 18,
    ):
        super().__init__()
        
        self.game_id = game_id
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=obs_shape,
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(num_actions)
        
        # Game-specific random behavior
        self._rng = np.random.RandomState(game_id)
        self._step_count = 0
        self._episode_reward = 0
        
        # Game-specific observation patterns
        self._obs_pattern = self._generate_game_pattern()
    
    def _generate_game_pattern(self) -> np.ndarray:
        """Generate a game-specific visual pattern."""
        # Each game has a unique visual signature
        pattern = np.zeros((84, 84), dtype=np.float32)
        
        # Add game-specific features
        for i in range(5):
            x = (self.game_id * 17 + i * 13) % 84
            y = (self.game_id * 23 + i * 7) % 84
            size = 5 + (self.game_id % 10)
            pattern[max(0, y-size):min(84, y+size), 
                   max(0, x-size):min(84, x+size)] = 128
        
        return pattern
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        
        self._step_count = 0
        self._episode_reward = 0
        
        obs = self._get_observation()
        return obs, {'game_id': self.game_id}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        
        # Game-specific reward function
        # Different games have different reward structures
        if action == self.game_id % self.num_actions:
            reward = 1.0
        else:
            reward = self._rng.random() * 0.1
        
        self._episode_reward += reward
        
        # Episode termination
        done = self._step_count >= 1000 or self._rng.random() < 0.001
        
        obs = self._get_observation()
        
        return obs, reward, done, False, {'game_id': self.game_id}
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation with game-specific patterns."""
        obs = np.zeros(self.obs_shape, dtype=np.uint8)
        
        # Base pattern
        base = self._obs_pattern + self._rng.randn(84, 84) * 20
        base = np.clip(base, 0, 255).astype(np.uint8)
        
        # Stack frames with slight variations
        for i in range(self.obs_shape[0]):
            frame = base + (self._step_count + i) % 50
            obs[i] = np.clip(frame, 0, 255).astype(np.uint8)
        
        return obs


def create_dummy_envs(num_games: int = 10) -> Tuple[list, list]:
    """Create dummy environments for testing."""
    envs = [DummyAtariEnv(game_id=i) for i in range(num_games)]
    names = [f"DummyGame_{i}" for i in range(num_games)]
    return envs, names

"""
Out-of-Core Mixture of Experts for Sequential Atari Learning

A hierarchical memory architecture that enables continual learning
across multiple Atari games through expert specialization and
meta-learned expert routing.

Key Components:
- MetaRSSM: Abstract world model for game-of-games dynamics
- Expert: Game-playing actor-critic networks
- ExpertManager: Orchestrates save/load/create with epsilon-threshold
- TieredStore: NVMe storage management
- PPOTrainer: Proximal Policy Optimization for experts
- GameCurriculum: Game sequence management
"""

from core import MetaRSSM, Expert, ExpertManager, TieredStore
from training import PPOTrainer
from envs import make_atari_env, GameCurriculum, CurriculumType

__version__ = "0.1.0"
__all__ = [
    "MetaRSSM",
    "Expert",
    "ExpertManager",
    "TieredStore",
    "PPOTrainer",
    "make_atari_env",
    "GameCurriculum",
    "CurriculumType",
]

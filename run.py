#!/usr/bin/env python3
"""
Main entry point for Out-of-Core MOE training.

Usage:
    python run.py --mode test           # Run tests
    python run.py --mode train          # Train on dummy envs
    python run.py --mode train_atari    # Train on real Atari (requires ALE)
"""

import argparse
import sys
import os
import torch
import numpy as np
from typing import List

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.moe_agent import MoERLAgent, MoERLAgentConfig
from training.ppo_trainer import SequentialGameTrainer
from envs.atari_wrappers import create_dummy_envs, make_atari_env, ATARI_GAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-Core MOE Training")
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="test",
        choices=["test", "train", "train_atari", "analyze"],
        help="Run mode"
    )
    
    # Model config
    parser.add_argument("--num_experts", type=int, default=256, help="Number of experts")
    parser.add_argument("--expert_dim", type=int, default=512, help="Expert dimension")
    parser.add_argument("--expert_hidden_dim", type=int, default=2048, help="Expert FFN hidden dim")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--top_k", type=int, default=2, help="Experts per token")
    parser.add_argument("--context_len", type=int, default=32, help="Context window length")
    
    # Memory hierarchy
    parser.add_argument("--hbm_capacity", type=int, default=32, help="Experts in HBM")
    parser.add_argument("--dram_capacity", type=int, default=128, help="Experts in DRAM")
    
    # Training
    parser.add_argument("--num_games", type=int, default=5, help="Number of games")
    parser.add_argument("--steps_per_game", type=int, default=100000, help="Steps per game")
    parser.add_argument("--num_rounds", type=int, default=1, help="Rounds through all games")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # System
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Log directory")
    
    return parser.parse_args()


def get_device(device_str: str) -> str:
    """Determine device to use."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_tests():
    """Run the test suite."""
    from tests.test_harness import run_all_tests
    success = run_all_tests()
    return 0 if success else 1


def train_dummy(args):
    """Train on dummy environments."""
    print("\n" + "=" * 60)
    print("Training on Dummy Environments")
    print("=" * 60)
    
    device = get_device(args.device)
    set_seed(args.seed)
    
    print(f"\nDevice: {device}")
    print(f"Seed: {args.seed}")
    
    # Create config
    config = MoERLAgentConfig(
        obs_shape=(1, 84, 84),
        num_actions=18,
        num_envs=args.num_games,
        frame_stack=4,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        context_len=args.context_len,
        hbm_capacity=args.hbm_capacity,
        dram_capacity=args.dram_capacity,
        learning_rate=args.lr,
    )
    
    # Print config
    param_counts = config.estimate_parameter_count()
    print(f"\nModel Configuration:")
    print(f"  Experts: {config.num_experts}")
    print(f"  Expert dim: {config.expert_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Total params: {param_counts['total']:,}")
    print(f"  Expert params: {param_counts['expert_params']:,}")
    print(f"  HBM capacity: {config.hbm_capacity} experts")
    print(f"  DRAM capacity: {config.dram_capacity} experts")
    
    # Create agent
    print("\nCreating agent...")
    agent = config.create_agent(device=device)
    
    # Create dummy environments
    print(f"Creating {args.num_games} dummy environments...")
    envs, env_names = create_dummy_envs(args.num_games)
    
    # Create trainer
    trainer = SequentialGameTrainer(
        agent=agent,
        config=config,
        envs=envs,
        env_names=env_names,
        steps_per_game=args.steps_per_game,
        log_dir=args.log_dir,
    )
    
    # Train
    print(f"\nStarting training for {args.num_rounds} round(s)...")
    print(f"Steps per game: {args.steps_per_game}")
    
    trainer.train(num_rounds=args.num_rounds)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    report = trainer.get_summary_report()
    
    print("\nFinal rewards per game:")
    for env_id, reward in report['final_rewards'].items():
        print(f"  {env_names[env_id]}: {reward:.2f}")
    
    print("\nCache statistics:")
    cache_stats = report['cache_stats']
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  Prefetch effectiveness: {cache_stats['prefetch_effectiveness']:.2%}")
    
    print("\nExpert specialization:")
    for env_id, spec in report['expert_specialization'].items():
        print(f"  {env_names[env_id]}: {spec['unique_experts_used']} unique experts, "
              f"concentration: {spec['concentration']:.2%}")
    
    if report['forgetting_metrics']:
        print("\nRetention on previous games:")
        for env_name, retentions in report['forgetting_metrics'].items():
            if retentions:
                print(f"  {env_name}: {retentions[-1]:.2%}")
    
    # Cleanup
    agent.expert_store.shutdown()
    
    return 0


def train_atari(args):
    """Train on real Atari environments."""
    print("\n" + "=" * 60)
    print("Training on Atari Games")
    print("=" * 60)
    
    try:
        import ale_py
        import gymnasium as gym
    except ImportError:
        print("\nError: Atari training requires ale-py and gymnasium[atari]")
        print("Install with: pip install gymnasium[atari] ale-py")
        return 1
    
    device = get_device(args.device)
    set_seed(args.seed)
    
    print(f"\nDevice: {device}")
    print(f"Seed: {args.seed}")
    
    # Select games
    games = ATARI_GAMES[:args.num_games]
    print(f"\nGames: {games}")
    
    # Create config
    config = MoERLAgentConfig(
        obs_shape=(1, 84, 84),
        num_actions=18,
        num_envs=len(games),
        frame_stack=4,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        expert_hidden_dim=args.expert_hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        context_len=args.context_len,
        hbm_capacity=args.hbm_capacity,
        dram_capacity=args.dram_capacity,
        learning_rate=args.lr,
    )
    
    # Create agent
    print("\nCreating agent...")
    agent = config.create_agent(device=device)
    
    # Create environments
    print("Creating Atari environments...")
    envs = []
    for game in games:
        env_id = f"ALE/{game}-v5"
        env = make_atari_env(env_id, frame_stack=4)
        envs.append(env)
    
    # Create trainer
    trainer = SequentialGameTrainer(
        agent=agent,
        config=config,
        envs=envs,
        env_names=games,
        steps_per_game=args.steps_per_game,
        log_dir=args.log_dir,
    )
    
    # Train
    print(f"\nStarting training...")
    trainer.train(num_rounds=args.num_rounds)
    
    # Print summary
    report = trainer.get_summary_report()
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    # Cleanup
    agent.expert_store.shutdown()
    for env in envs:
        env.close()
    
    return 0


def analyze_experts(args):
    """Analyze expert specialization patterns."""
    print("\n" + "=" * 60)
    print("Expert Specialization Analysis")
    print("=" * 60)
    
    # This would load a checkpoint and analyze
    print("\nAnalysis mode not yet implemented.")
    print("This would load a trained model and analyze:")
    print("  - Which experts specialize to which games")
    print("  - Expert co-occurrence patterns")
    print("  - Cache efficiency over training")
    
    return 0


def main():
    args = parse_args()
    
    if args.mode == "test":
        return run_tests()
    elif args.mode == "train":
        return train_dummy(args)
    elif args.mode == "train_atari":
        return train_atari(args)
    elif args.mode == "analyze":
        return analyze_experts(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

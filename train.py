"""
Main training orchestrator for Out-of-Core MoE on Atari.

Implements the bi-level training loop:
- Day phase: K games with expert training
- Night phase: Meta-agent update

The meta-agent detects game changes, manages expert swapping,
and learns to prefetch based on game transition patterns.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import time
from tqdm import tqdm

from core import MetaRSSM, Expert, ExpertManager
from training import PPOTrainer
from envs import make_atari_env, GameCurriculum, CurriculumType, ATARI_GAMES
from envs.action_space import UnifiedActionSpace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DayNightTrainer:
    """
    Bi-level training loop for hierarchical MoE.

    Day phase: Play K games, train experts, update meta-agent (unsupervised)
    Night phase: Update meta-agent with cumulative expert performance
    """

    def __init__(
        self,
        games: List[str],
        curriculum_type: CurriculumType = CurriculumType.MARKOV,
        device: torch.device = torch.device("cuda"),
        # Meta-agent params
        meta_lr: float = 1e-4,
        kl_threshold: float = 2.0,
        # Expert params
        expert_lr: float = 2.5e-4,
        n_steps_per_update: int = 128,
        # Training params
        games_per_day: int = 5,
        episodes_per_game: int = 3,
        num_days: int = 100,
        # Storage
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        seed: Optional[int] = None,
    ):
        self.device = device
        self.games = games
        self.games_per_day = games_per_day
        self.episodes_per_game = episodes_per_game
        self.num_days = num_days
        self.seed = seed

        # Paths
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir)

        # Curriculum
        self.curriculum = GameCurriculum(
            games=games,
            curriculum_type=curriculum_type,
            seed=seed,
        )

        # Meta-agent
        self.meta_agent = MetaRSSM(
            obs_shape=(4, 84, 84),
            hidden_dim=256,
            code_dim=32,
            codebook_size=64,
            kl_threshold=kl_threshold,
        ).to(device)

        self.meta_optimizer = Adam(self.meta_agent.parameters(), lr=meta_lr)

        # Unified action space across all training games
        self.unified_action_space = UnifiedActionSpace(games)
        self.unified_action_space.to_device(device)

        # Expert manager
        self.expert_manager = ExpertManager(
            storage_path=str(self.save_dir / "experts"),
            device=device,
            epsilon=0.5,
            code_dim=32,
            codebook_size=64,
            num_actions=self.unified_action_space.num_actions,
            unified_action_space=self.unified_action_space,
        )

        # PPO trainer (will be set per expert)
        self.ppo_trainer: Optional[PPOTrainer] = None
        self.expert_lr = expert_lr
        self.n_steps_per_update = n_steps_per_update

        # Statistics
        self.global_step = 0
        self.total_episodes = 0
        self.day_count = 0

    def _create_env(self, game_name: str):
        """Create wrapped Atari environment."""
        return make_atari_env(game_name, seed=self.seed)

    def _init_obs(self, env) -> torch.Tensor:
        """Get initial observation as tensor."""
        obs, _ = env.reset()
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def run_day(self) -> Dict:
        """
        Run one day of training: K games with expert training.

        Returns:
            Day statistics including selection_log_probs for REINFORCE
        """
        day_rewards = []
        day_frames = 0
        meta_losses = []
        selection_data = []  # Track (h_state, code_idx) for REINFORCE

        # Meta-agent state
        meta_state = self.meta_agent.init_state(1, self.device)

        for game_idx in range(self.games_per_day):
            game_name = self.curriculum.get_current_game()
            logger.info(f"Day {self.day_count}, Game {game_idx + 1}/{self.games_per_day}: {game_name}")

            # Create environment
            env = self._create_env(game_name)
            obs = self._init_obs(env)

            # Get game embedding from meta-agent
            meta_state, meta_outputs = self.meta_agent(obs, meta_state)
            game_embedding = self.meta_agent.get_game_embedding(meta_state)
            game_code = self.meta_agent.get_game_code(meta_state)  # Keep as tensor

            # Store (h_state, code_idx) for REINFORCE - detach h to avoid graph issues
            # We'll recompute log_probs at night with current parameters
            selection_data.append((meta_state.h.detach().clone(), game_code.detach().clone()))

            game_code_int = game_code.item()  # Convert to int for expert manager

            # Detect switch and get appropriate expert
            kl = meta_outputs['kl'].item()
            switch_detected = kl > self.meta_agent.kl_threshold

            if switch_detected or self.expert_manager.active_expert is None:
                logger.info(f"Switch detected (KL={kl:.2f}), retrieving expert")
                expert = self.expert_manager.retrieve_or_create(
                    embedding=game_embedding.squeeze(0),
                    code_idx=game_code_int,
                )
            else:
                expert = self.expert_manager.get_active_expert()

            # Setup PPO trainer for this expert
            if self.ppo_trainer is None or self.ppo_trainer.expert != expert:
                self.ppo_trainer = PPOTrainer(
                    expert=expert,
                    device=self.device,
                    learning_rate=self.expert_lr,
                    n_steps=self.n_steps_per_update,
                    unified_action_space=self.unified_action_space,
                )
            else:
                self.ppo_trainer.set_expert(expert)

            # Play episodes with this expert
            game_rewards = []
            for ep_idx in range(self.episodes_per_game):
                obs = self._init_obs(env)
                episode_reward = 0
                episode_length = 0
                done = False

                while not done:
                    # Update meta-agent state
                    meta_state, meta_outputs = self.meta_agent(obs, meta_state)

                    # Check for mid-game switch (shouldn't happen often)
                    kl = meta_outputs['kl'].item()

                    # Collect rollout and train
                    obs, metrics = self.ppo_trainer.step(env, obs, game_name=game_name)

                    # Accumulate rewards from rollout info
                    episode_rewards = metrics.get('episode_rewards', [])
                    if episode_rewards:
                        game_rewards.extend(episode_rewards)
                        self.total_episodes += len(episode_rewards)

                    day_frames += self.n_steps_per_update

                    # Check if episode ended in rollout
                    done = len(episode_rewards) > 0

                    # Log training metrics
                    self.global_step += self.n_steps_per_update
                    self.writer.add_scalar('train/policy_loss', metrics['policy_loss'], self.global_step)
                    self.writer.add_scalar('train/value_loss', metrics['value_loss'], self.global_step)
                    self.writer.add_scalar('train/entropy', -metrics['entropy_loss'], self.global_step)
                    self.writer.add_scalar('meta/kl', kl, self.global_step)

            # Update expert metadata
            expert.update_metadata(
                game_name=game_name,
                frames=day_frames,
                episodes=len(game_rewards),
                episode_reward=max(game_rewards) if game_rewards else None,
            )

            # Update expert centroid
            self.expert_manager.update_centroid(
                expert.expert_id,
                game_embedding.squeeze(0),
            )

            # Meta-agent unsupervised learning on trajectory
            meta_loss, meta_metrics = self._train_meta_step(obs)
            meta_losses.append(meta_loss)

            day_rewards.extend(game_rewards)

            # Next game
            self.curriculum.next_game()

            # Prefetch next likely expert
            trans_dist = self.meta_agent.get_prefetch_distribution(meta_state)
            self.expert_manager.prefetch_from_distribution(trans_dist.squeeze(0))

            env.close()

        return {
            'day_reward': sum(day_rewards),
            'mean_episode_reward': np.mean(day_rewards) if day_rewards else 0,
            'num_episodes': len(day_rewards),
            'day_frames': day_frames,
            'mean_meta_loss': np.mean(meta_losses),
            'selection_data': selection_data,  # For REINFORCE
        }

    def _train_meta_step(self, obs: torch.Tensor) -> tuple:
        """
        Single meta-agent training step (unsupervised).

        Uses recent observation to update the RSSM.
        """
        self.meta_optimizer.zero_grad()

        # Create mini-trajectory from current observation
        # In practice, you'd accumulate a buffer of observations
        trajectory = obs.unsqueeze(1).repeat(1, 8, 1, 1, 1)  # (1, 8, C, H, W)

        loss, metrics = self.meta_agent.compute_loss(trajectory)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.meta_agent.parameters(), 1.0)
        self.meta_optimizer.step()

        return loss.item(), metrics

    def run_night(self, day_stats: Dict) -> Dict:
        """
        Run night phase: meta-agent REINFORCE update based on day performance.

        Args:
            day_stats: Statistics from the day phase (includes selection_log_probs)

        Returns:
            Night update metrics
        """
        cumulative_reward = day_stats['day_reward']
        selection_data = day_stats.get('selection_data', [])

        # Compute REINFORCE loss and get metrics
        self.meta_optimizer.zero_grad()
        reinforce_loss, metrics = self.meta_agent.night_update(
            cumulative_reward,
            selection_data=selection_data,
        )

        # Apply REINFORCE gradient if loss was computed
        if reinforce_loss is not None:
            reinforce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_agent.parameters(), 1.0)
            self.meta_optimizer.step()

        # Logging
        self.writer.add_scalar('night/cumulative_reward', cumulative_reward, self.day_count)
        self.writer.add_scalar('night/baseline', metrics['baseline'], self.day_count)
        self.writer.add_scalar('night/advantage', metrics['advantage'], self.day_count)
        if 'reinforce_loss' in metrics:
            self.writer.add_scalar('night/reinforce_loss', metrics['reinforce_loss'], self.day_count)
        if 'mean_log_prob' in metrics:
            self.writer.add_scalar('night/mean_log_prob', metrics['mean_log_prob'], self.day_count)

        return metrics

    def train(self):
        """Run full training loop."""
        logger.info(f"Starting training for {self.num_days} days")
        logger.info(f"Games per day: {self.games_per_day}")
        logger.info(f"Episodes per game: {self.episodes_per_game}")
        logger.info(f"Games: {self.games}")

        pbar = tqdm(range(self.num_days), desc="Training")

        for day in pbar:
            self.day_count = day

            # Day phase
            day_stats = self.run_day()

            # Night phase
            night_stats = self.run_night(day_stats)

            # Logging
            self.writer.add_scalar('day/reward', day_stats['day_reward'], day)
            self.writer.add_scalar('day/mean_episode_reward', day_stats['mean_episode_reward'], day)
            self.writer.add_scalar('day/episodes', day_stats['num_episodes'], day)
            self.writer.add_scalar('day/frames', day_stats['day_frames'], day)

            pbar.set_postfix({
                'reward': f"{day_stats['mean_episode_reward']:.1f}",
                'episodes': day_stats['num_episodes'],
                'experts': len(self.expert_manager.list_experts()),
            })

            # Save checkpoint periodically
            if (day + 1) % 10 == 0:
                self.save_checkpoint(day)

        # Final save
        self.save_checkpoint(self.num_days)
        self.expert_manager.save_all()

        logger.info("Training complete!")
        logger.info(f"Total experts created: {len(self.expert_manager.list_experts())}")

    def save_checkpoint(self, day: int):
        """Save training checkpoint."""
        checkpoint = {
            'day': day,
            'meta_agent': self.meta_agent.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'global_step': self.global_step,
            'total_episodes': self.total_episodes,
        }
        path = self.save_dir / f"checkpoint_day{day:04d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_agent.load_state_dict(checkpoint['meta_agent'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.day_count = checkpoint['day']
        self.global_step = checkpoint['global_step']
        self.total_episodes = checkpoint['total_episodes']
        logger.info(f"Loaded checkpoint from {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Out-of-Core MoE on Atari")
    parser.add_argument("--games", nargs="+", default=["Breakout", "Pong", "SpaceInvaders"])
    parser.add_argument("--curriculum", choices=["random", "markov", "periodic"], default="markov")
    parser.add_argument("--num-days", type=int, default=100)
    parser.add_argument("--games-per-day", type=int, default=5)
    parser.add_argument("--episodes-per-game", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--log-dir", default="./logs")
    args = parser.parse_args()

    curriculum_type = {
        "random": CurriculumType.RANDOM,
        "markov": CurriculumType.MARKOV,
        "periodic": CurriculumType.PERIODIC,
    }[args.curriculum]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    trainer = DayNightTrainer(
        games=args.games,
        curriculum_type=curriculum_type,
        device=device,
        num_days=args.num_days,
        games_per_day=args.games_per_day,
        episodes_per_game=args.episodes_per_game,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )

    trainer.train()


if __name__ == "__main__":
    main()

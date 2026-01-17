from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from envs import make_atari_env
from models import MoEDQN
from replay import SleepBuffer, Transition
from utils.device import get_default_device
from utils.seed import set_global_seeds
from .ewc import EWC, estimate_fisher_diag, make_snapshot


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    t = min(max(step, 0), decay_steps) / float(decay_steps)
    return float(start + t * (end - start))


@dataclass
class DayStats:
    episode_returns: List[float]
    steps: int
    cache_hit_rate: float
    expert_scores: np.ndarray  # (num_experts,)


class SimpleExpertCacheSimulator:
    """A tiny cache simulator for tracking 'HBM resident' experts.

    This does NOT move parameters. It just tracks reuse/hit-rate, which is useful
    before implementing real HBM/DRAM/NVMe swapping.
    """

    def __init__(self, capacity_experts: int):
        self.capacity = int(capacity_experts)
        self.resident: List[int] = []
        self.hits = 0
        self.misses = 0

    def prefetch(self, expert_ids: List[int]) -> None:
        # Simple: replace the cache content with the prefetched experts (unique, truncated)
        uniq = []
        for e in expert_ids:
            if e not in uniq:
                uniq.append(int(e))
        self.resident = uniq[: self.capacity]

    def access(self, expert_ids: List[int]) -> None:
        for e in expert_ids:
            e = int(e)
            if e in self.resident:
                self.hits += 1
            else:
                self.misses += 1
                # LRU-ish insert
                self.resident.insert(0, e)
                self.resident = self.resident[: self.capacity]

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits) / float(total) if total > 0 else 0.0

    def reset_counts(self) -> None:
        self.hits = 0
        self.misses = 0


class DayNightTrainer:
    def __init__(
        self,
        games: List[str],
        cfg: Config,
        *,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.games = list(games)
        self.cfg = cfg
        self.seed = int(seed)
        set_global_seeds(self.seed)

        self.device = device or get_default_device()

        # We'll assume Atari with unified full action space => Discrete(18)
        self.action_dim = 18

        self.model = MoEDQN(
            obs_stack=self.cfg.frame_stack,
            action_dim=self.action_dim,
            num_experts=self.cfg.num_experts,
            router_hidden_dim=self.cfg.router_hidden_dim,
            expert_hidden_dim=self.cfg.expert_hidden_dim,
            feature_dim=self.cfg.feature_dim,
            temperature=1.0,
        ).to(self.device)

        self.target_model = MoEDQN(
            obs_stack=self.cfg.frame_stack,
            action_dim=self.action_dim,
            num_experts=self.cfg.num_experts,
            router_hidden_dim=self.cfg.router_hidden_dim,
            expert_hidden_dim=self.cfg.expert_hidden_dim,
            feature_dim=self.cfg.feature_dim,
            temperature=1.0,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.ewc = EWC(lambda_=self.cfg.ewc_lambda)

        self.sleep_buffer = SleepBuffer()
        self.global_step = 0

        # Per-game routing statistics
        self.flagged_experts: Dict[str, List[int]] = {}

    # ------------------------ acting (day) ------------------------

    @torch.no_grad()
    def _select_action(self, q: torch.Tensor, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        return int(torch.argmax(q, dim=-1).item())

    @staticmethod
    def _top_experts_from_gating(gating_probs: torch.Tensor, k: int = 2) -> List[int]:
        # gating_probs: (E,)
        vals, idx = torch.topk(gating_probs, k=min(k, gating_probs.shape[-1]))
        return [int(i) for i in idx.tolist()]

    def run_day_for_game(self, game_id: str, *, steps: int, cache_capacity: int = 4) -> DayStats:
        env = make_atari_env(
            game_id,
            seed=self.seed,
            frame_stack=self.cfg.frame_stack,
            clip_rewards=self.cfg.reward_clip,
            full_action_space=True,
        )

        obs, info = env.reset()
        hidden = self.model.init_hidden(1, self.device)

        # Prefetch last known flagged experts for this game (if any)
        cache = SimpleExpertCacheSimulator(capacity_experts=cache_capacity)
        cache.prefetch(self.flagged_experts.get(game_id, []))

        episode_return = 0.0
        episode_returns: List[float] = []

        expert_scores = np.zeros((self.cfg.num_experts,), dtype=np.float32)

        for t in range(int(steps)):
            eps = linear_epsilon(self.global_step, self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps)

            obs_t = torch.from_numpy(obs).to(self.device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)  # (1, stack, 84, 84)

            out = self.model(obs_t, hidden)
            q = out.q_values  # (1, A)
            gating = out.gating_probs.squeeze(0)  # (E,)

            # Access top-2 experts for cache stats
            top2 = self._top_experts_from_gating(gating, k=2)
            cache.access(top2)

            action = self._select_action(q.squeeze(0), eps)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Salience estimate (cheap): TD-error proxy + action surprisal
            # TD-error uses target network bootstrap.
            td_err, surprisal = self._compute_salience_components(obs, action, reward, next_obs, done, hidden)
            salience = float(self.cfg.td_error_weight * td_err + self.cfg.policy_surprisal_weight * surprisal)

            # Update expert usage stats (responsibility * salience)
            with torch.no_grad():
                expert_scores += gating.detach().cpu().numpy() * salience

            # Store transition in day buffer
            self.sleep_buffer.add(
                game_id,
                Transition(
                    obs=np.array(obs, copy=True),
                    action=int(action),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    next_obs=np.array(next_obs, copy=True),
                    salience=max(1e-6, float(abs(salience))),
                ),
            )

            episode_return += float(reward)

            # Update hidden state for next step
            hidden = out.hidden.detach()
            if done:
                self.sleep_buffer.end_episode(game_id)
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, info = env.reset()
                hidden = self.model.init_hidden(1, self.device)
            else:
                obs = next_obs

            self.global_step += 1

        env.close()

        # End any partial episode
        self.sleep_buffer.end_episode(game_id)

        return DayStats(
            episode_returns=episode_returns,
            steps=int(steps),
            cache_hit_rate=cache.hit_rate(),
            expert_scores=expert_scores,
        )

    @torch.no_grad()
    def _compute_salience_components(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        hidden: torch.Tensor,
    ) -> Tuple[float, float]:
        """Returns (abs TD error, action surprisal)."""
        obs_t = torch.from_numpy(obs).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
            next_obs_t = next_obs_t.unsqueeze(0)

        out = self.model(obs_t, hidden)
        q = out.q_values  # (1, A)

        # Surprisal using softmax over Q-values (approx policy)
        temp = max(self.cfg.softmax_temp_for_surprisal, 1e-6)
        logp = F.log_softmax(q / temp, dim=-1)
        surprisal = float((-logp[0, int(action)]).detach().cpu().item())

        # TD error
        q_sa = float(q[0, int(action)].detach().cpu().item())

        # Bootstrap
        q_next, _ = self._target_q_next(next_obs_t, out.hidden)
        max_next = float(torch.max(q_next, dim=-1).values.detach().cpu().item())
        target = float(reward) + (0.0 if done else self.cfg.gamma * max_next)
        td_err = abs(target - q_sa)

        return float(td_err), float(surprisal)

    @torch.no_grad()
    def _target_q_next(self, next_obs_t: torch.Tensor, hidden_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_next = self.target_model(next_obs_t, hidden_t)
        return out_next.q_values, out_next.hidden

    # ------------------------ learning (sleep) ------------------------

    def _loss_on_sequence_batch(self, model: MoEDQN, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DRQN-style DQN loss on a (B, T, ...) batch."""
        obs = batch["obs"]            # (B, T, C, H, W)
        actions = batch["actions"]    # (B, T)
        rewards = batch["rewards"]    # (B, T)
        dones = batch["dones"]        # (B, T)
        next_obs = batch["next_obs"]  # (B, T, C, H, W)
        salience = batch["salience"]  # (B, T)

        B, T = actions.shape

        hidden = model.init_hidden(B, obs.device)

        total_loss = torch.zeros((), device=obs.device)

        for t in range(T):
            out = model(obs[:, t], hidden)
            q = out.q_values
            hidden = out.hidden

            # Reset hidden where done (to avoid leaking context across episode boundaries)
            done_t = dones[:, t].view(B, 1)
            hidden = hidden * (1.0 - done_t)

            # Q(s,a)
            a_t = actions[:, t].long().view(B, 1)
            q_sa = q.gather(1, a_t).squeeze(1)

            with torch.no_grad():
                q_next, _ = self._target_q_next(next_obs[:, t], hidden)
                max_next = q_next.max(dim=-1).values
                target = rewards[:, t] + (1.0 - dones[:, t]) * self.cfg.gamma * max_next

            # Huber
            td = q_sa - target
            huber = torch.where(td.abs() < 1.0, 0.5 * td.pow(2), td.abs() - 0.5)

            # Salience-weighted loss (acts like prioritization if sampling is imperfect)
            w = (salience[:, t].detach() + 1e-6) ** self.cfg.salience_alpha
            w = w / (w.mean() + 1e-6)

            total_loss = total_loss + (w * huber).mean()

        # Add EWC penalty
        total_loss = total_loss + self.ewc.penalty(model)
        return total_loss

    def _iter_fisher_batches(self, game_id: str, *, max_batches: int) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield a few batches for Fisher estimation."""
        rng = np.random.default_rng(self.seed)
        for _ in range(max_batches):
            sample = self.sleep_buffer.sample_sequences(
                game_id,
                batch_size=self.cfg.batch_size,
                seq_len=self.cfg.seq_len,
                prioritize=True,
                alpha=self.cfg.salience_alpha,
                rng=rng,
            )
            yield self._to_torch_batch(sample)

    def _to_torch_batch(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        # Observations from gymnasium are (stack, 84,84). In batch we want (B,T,stack,84,84)
        obs = torch.from_numpy(sample["obs"]).to(self.device)
        next_obs = torch.from_numpy(sample["next_obs"]).to(self.device)

        # Ensure channel-first for conv encoder: (B,T,C,H,W)
        if obs.ndim == 5:
            pass
        else:
            raise RuntimeError(f"Unexpected obs shape: {obs.shape}")

        return {
            "obs": obs,
            "actions": torch.from_numpy(sample["actions"]).to(self.device),
            "rewards": torch.from_numpy(sample["rewards"]).to(self.device),
            "dones": torch.from_numpy(sample["dones"]).to(self.device),
            "next_obs": next_obs,
            "salience": torch.from_numpy(sample["salience"]).to(self.device),
        }

    def run_sleep(self, games_today: List[str]) -> Dict[str, float]:
        """Run the sleep phase training on today's games only."""
        metrics: Dict[str, float] = {}
        updates = 0

        # Simple block schedule: loop over games, do N updates each
        for game_id in games_today:
            if self.sleep_buffer.num_transitions(game_id) < self.cfg.seq_len:
                continue

            for _ in range(self.cfg.sleep_updates_per_game):
                sample = self.sleep_buffer.sample_sequences(
                    game_id,
                    batch_size=self.cfg.batch_size,
                    seq_len=self.cfg.seq_len,
                    prioritize=True,
                    alpha=self.cfg.salience_alpha,
                )
                batch = self._to_torch_batch(sample)

                self.optim.zero_grad(set_to_none=True)
                loss = self._loss_on_sequence_batch(self.model, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()

                updates += 1

                if updates % self.cfg.target_update_interval == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # After training on this game in sleep, estimate fisher and snapshot for EWC.
            fisher_filter = self._make_ewc_filter_fn(game_id)
            fisher = estimate_fisher_diag(
                self.model,
                batches=self._iter_fisher_batches(game_id, max_batches=self.cfg.fisher_batches),
                loss_fn=lambda m, b: self._loss_on_sequence_batch(m, b) - self.ewc.penalty(m),
                filter_fn=fisher_filter,
                max_batches=self.cfg.fisher_batches,
            )
            snapshot = make_snapshot(self.model, name=game_id, fisher_diag=fisher, filter_fn=fisher_filter)
            self.ewc.add_task_snapshot(snapshot)

            metrics[f"sleep_updates_{game_id}"] = float(self.cfg.sleep_updates_per_game)

        metrics["sleep_total_updates"] = float(updates)
        return metrics

    def _make_ewc_filter_fn(self, game_id: str):
        flagged = set(self.flagged_experts.get(game_id, []))

        def keep(name: str) -> bool:
            # Always protect router
            if name.startswith("router."):
                return True
            if self.cfg.protect_encoder and name.startswith("encoder."):
                return True
            if self.cfg.protect_experts and name.startswith("experts."):
                # experts.<idx>.
                try:
                    parts = name.split(".")
                    idx = int(parts[1])
                    return idx in flagged
                except Exception:
                    return False
            return False

        return keep

    # ------------------------ full cycle ------------------------

    def run_one_day(self, day_index: int) -> Dict[str, object]:
        """Runs one day: pick K games, play them sequentially, then sleep train, then flush."""
        rng = random.Random(self.seed + day_index)
        games_today = rng.sample(self.games, k=min(self.cfg.games_per_day, len(self.games)))

        day_summaries: Dict[str, DayStats] = {}
        for g in games_today:
            stats = self.run_day_for_game(g, steps=self.cfg.day_steps_per_game)
            day_summaries[g] = stats

            # Update flagged experts for this game based on today's usage
            topk = int(self.cfg.top_experts_per_game)
            top_ids = np.argsort(-stats.expert_scores)[:topk].tolist()
            self.flagged_experts[g] = [int(i) for i in top_ids]

        sleep_metrics = self.run_sleep(games_today)

        # Flush sleep buffer at end of sleep cycle (per your spec)
        self.sleep_buffer.flush()

        out: Dict[str, object] = {
            "day": day_index,
            "games_today": games_today,
            "episode_returns": {g: day_summaries[g].episode_returns for g in games_today},
            "cache_hit_rate": {g: day_summaries[g].cache_hit_rate for g in games_today},
            "flagged_experts": {g: self.flagged_experts[g] for g in games_today},
            "sleep": sleep_metrics,
            "ewc_tasks": len(self.ewc.tasks),
        }
        return out

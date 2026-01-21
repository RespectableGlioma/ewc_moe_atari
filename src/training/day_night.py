from __future__ import annotations

import random
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from envs import make_atari_env
from models import MoEDQN
from replay import SleepBuffer, Transition
from store.expert_store import ExpertStore, ExpertStoreStats
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
    expert_scores: np.ndarray  # (num_experts,)
    store_stats: ExpertStoreStats


class DayNightTrainer:
    def __init__(
        self,
        games: List[str],
        cfg: Config,
        *,
        seed: int = 0,
        device: Optional[torch.device] = None,
        run_dir: Optional[str | Path] = None,
    ):
        self.games = list(games)
        self.cfg = cfg
        self.seed = int(seed)
        set_global_seeds(self.seed)

        self.device = device or get_default_device()
        self.run_dir = Path(run_dir) if run_dir is not None else None

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

        # Expert stores (online + target)
        base = self.run_dir if self.run_dir is not None else Path("./runs/_tmp")
        experts_dir_online = base / "experts" / "online"
        experts_dir_target = base / "experts" / "target"

        self.expert_store = ExpertStore(
            experts=self.model.experts,
            disk_dir=experts_dir_online,
            hbm_capacity=self.cfg.hbm_expert_capacity,
            dram_capacity=self.cfg.dram_expert_capacity,
            device=self.device,
            enable_disk=self.cfg.enable_nvme_tier,
            pin_cpu=self.cfg.pin_cpu_memory,
            create_expert_optimizer=True,
            expert_lr=self.cfg.learning_rate,
        )

        self.target_store = ExpertStore(
            experts=self.target_model.experts,
            disk_dir=experts_dir_target,
            hbm_capacity=self.cfg.hbm_expert_capacity,
            dram_capacity=self.cfg.dram_expert_capacity,
            device=self.device,
            enable_disk=self.cfg.enable_nvme_tier,
            pin_cpu=self.cfg.pin_cpu_memory,
            create_expert_optimizer=False,
        )

        # Cold-start experts to disk to force real paging behavior
        if self.cfg.enable_nvme_tier:
            self.expert_store.cold_start_to_disk()
            self.target_store.cold_start_to_disk()

        # Core optimizer (router + encoder). Experts are optimized via per-expert optimizers.
        core_params = list(self.model.encoder.parameters()) + list(self.model.router.parameters())
        self.core_optim = torch.optim.Adam(core_params, lr=self.cfg.learning_rate)

        self.ewc = EWC(lambda_=self.cfg.ewc_lambda)
        self.sleep_buffer = SleepBuffer()
        self.global_step = 0

        # Per-game routing statistics
        self.flagged_experts: Dict[str, List[int]] = {}

        # Track which experts were updated since last target sync
        self._dirty_experts: Set[int] = set()

        # Track which experts were updated since the start of the current night (for NVMe writeback).
        self._dirty_since_night: Set[int] = set()

        # A tiny generic 'base' expert set kept warm across all days (helps act immediately).
        base_n = int(getattr(self.cfg, 'base_warm_experts', 2))
        base_n = max(0, min(base_n, int(self.cfg.num_experts)))
        # Don't exceed HBM capacity when we prefetch the base set.
        base_n = min(base_n, int(self.cfg.hbm_expert_capacity))
        self.base_experts: List[int] = list(range(base_n))

        # Warm DRAM set retained across days (capacity-limited).
        self.warm_experts: List[int] = list(self.base_experts)

        # Decayed retention scores used to pick the warm set each night.
        self._retain_scores = np.zeros((self.cfg.num_experts,), dtype=np.float32)

        # Ensure base experts are materialized on CPU so day-0 can act immediately after a cold start.
        if self.cfg.enable_nvme_tier and self.base_experts:
            self.expert_store.ensure_on_cpu(self.base_experts)
            self.target_store.ensure_on_cpu(self.base_experts)


        # Optional checkpoint dir
        if self.run_dir is not None:
            (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ------------------------ acting (day) ------------------------

    @torch.no_grad()
    def _select_action(self, q: torch.Tensor, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        return int(torch.argmax(q, dim=-1).item())

    def run_day_for_game(self, game_id: str, *, steps: int) -> DayStats:
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
        pref = self.flagged_experts.get(game_id, [])
        if pref:
            self.expert_store.prefetch_to_gpu(pref)
            self.target_store.prefetch_to_gpu(pref)

        self.expert_store.reset_stats()

        episode_return = 0.0
        episode_returns: List[float] = []

        expert_scores = np.zeros((self.cfg.num_experts,), dtype=np.float32)

        for _ in range(int(steps)):
            eps = linear_epsilon(self.global_step, self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps)

            obs_t = torch.from_numpy(obs).to(self.device)
            if obs_t.ndim == 3:
                obs_t = obs_t.unsqueeze(0)  # (1, stack, 84, 84)

            out = self.model(
                obs_t,
                hidden,
                expert_store=self.expert_store,
                top_k=self.cfg.expert_top_k,
            )
            q = out.q_values  # (1, A)
            gating = out.gating_probs.squeeze(0)  # (E,)

            action = self._select_action(q.squeeze(0), eps)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Salience estimate (cheap-ish): TD-error proxy + action surprisal
            td_err, surprisal = self._compute_salience_components(obs, action, reward, next_obs, done, hidden)
            salience = float(self.cfg.td_error_weight * td_err + self.cfg.policy_surprisal_weight * surprisal)

            # Update expert usage stats (responsibility * salience)
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

        store_stats = self.expert_store.reset_stats()
        return DayStats(
            episode_returns=episode_returns,
            steps=int(steps),
            expert_scores=expert_scores,
            store_stats=store_stats,
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

        out = self.model(obs_t, hidden, expert_store=self.expert_store, top_k=self.cfg.expert_top_k)
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
        out_next = self.target_model(next_obs_t, hidden_t, expert_store=self.target_store, top_k=self.cfg.expert_top_k)
        return out_next.q_values, out_next.hidden

    # ------------------------ learning (sleep) ------------------------

    def _loss_on_sequence_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        used_experts: Set[int],
        include_ewc: bool = True,
        include_moe_aux: bool = True,
    ) -> torch.Tensor:
        """Compute DRQN-style DQN loss on a (B, T, ...) batch."""
        obs = batch["obs"]            # (B, T, C, H, W)
        actions = batch["actions"]    # (B, T)
        rewards = batch["rewards"]    # (B, T)
        dones = batch["dones"]        # (B, T)
        next_obs = batch["next_obs"]  # (B, T, C, H, W)
        salience = batch["salience"]  # (B, T)

        B, T = actions.shape

        # Optional router auxiliary losses to encourage expert diversity / reduce collapse.
        entropy_coef = float(getattr(self.cfg, 'router_entropy_coef', getattr(self.cfg, 'router_entropy_weight', 0.0)))
        balance_coef = float(getattr(self.cfg, 'router_load_balance_coef', getattr(self.cfg, 'router_balance_coef', 0.0)))
        aux_entropy = torch.zeros((), device=self.device)
        aux_kl = torch.zeros((), device=self.device)
        aux_n = 0


        hidden = self.model.init_hidden(B, obs.device)
        total_loss = torch.zeros((), device=obs.device)

        for t in range(T):
            out = self.model(
                obs[:, t],
                hidden,
                expert_store=self.expert_store,
                top_k=self.cfg.expert_top_k,
                record_used_experts=used_experts,
            )
            q = out.q_values
            hidden = out.hidden

            # Router aux stats (gating probabilities) for load balancing / entropy regularization
            if include_moe_aux and (entropy_coef > 0.0 or balance_coef > 0.0):
                g = getattr(out, 'gating_probs', None)
                if g is not None:
                    eps = 1e-8
                    g = g / (g.sum(dim=-1, keepdim=True) + eps)
                    ent = -(g * (g + eps).log()).sum(dim=-1).mean()
                    aux_entropy = aux_entropy + ent
                    mean_g = g.mean(dim=0)
                    mean_g = mean_g / (mean_g.sum() + eps)
                    uniform = torch.full_like(mean_g, 1.0 / mean_g.numel())
                    kl = (mean_g * ((mean_g + eps).log() - (uniform + eps).log())).sum()
                    aux_kl = aux_kl + kl
                    aux_n += 1


            # Reset hidden where done
            done_t = dones[:, t].view(B, 1)
            hidden = hidden * (1.0 - done_t)

            a_t = actions[:, t].long().view(B, 1)
            q_sa = q.gather(1, a_t).squeeze(1)

            with torch.no_grad():
                q_next, _ = self._target_q_next(next_obs[:, t], hidden)
                max_next = q_next.max(dim=-1).values
                target = rewards[:, t] + (1.0 - dones[:, t]) * self.cfg.gamma * max_next

            td = q_sa - target
            huber = torch.where(td.abs() < 1.0, 0.5 * td.pow(2), td.abs() - 0.5)

            w = (salience[:, t].detach() + 1e-6) ** self.cfg.salience_alpha
            w = w / (w.mean() + 1e-6)

            total_loss = total_loss + (w * huber).mean()


        # Router auxiliary regularizers (optional).
        # - Entropy bonus: encourages per-step gating distributions to stay high-entropy (less collapse).
        # - Load-balance KL: encourages the *average* gating distribution to be closer to uniform.
        #
        # Both are crude but effective first-order anti-collapse terms for research scaffolds.
        if include_moe_aux and aux_n > 0:
            if entropy_coef > 0.0:
                total_loss = total_loss - entropy_coef * (aux_entropy / float(aux_n))
            if balance_coef > 0.0:
                total_loss = total_loss + balance_coef * (aux_kl / float(aux_n))

        if include_ewc:
            total_loss = total_loss + self.ewc.penalty(self.model)
        return total_loss

    def _iter_fisher_batches(self, game_id: str, *, max_batches: int) -> Iterator[Dict[str, torch.Tensor]]:
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
        obs = torch.from_numpy(sample["obs"]).to(self.device)
        next_obs = torch.from_numpy(sample["next_obs"]).to(self.device)

        if obs.ndim != 5:
            raise RuntimeError(f"Unexpected obs shape: {obs.shape}")

        return {
            "obs": obs,
            "actions": torch.from_numpy(sample["actions"]).to(self.device),
            "rewards": torch.from_numpy(sample["rewards"]).to(self.device),
            "dones": torch.from_numpy(sample["dones"]).to(self.device),
            "next_obs": next_obs,
            "salience": torch.from_numpy(sample["salience"]).to(self.device),
        }

    def _make_ewc_filter_fn(self, game_id: str):
        flagged = set(self.flagged_experts.get(game_id, []))

        def keep(name: str) -> bool:
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

    def _sync_target(self) -> None:
        """Sync encoder/router always, and sync experts that were recently updated."""
        # Core
        self.target_model.encoder.load_state_dict(self.model.encoder.state_dict())
        self.target_model.router.load_state_dict(self.model.router.state_dict())

        # Experts: only sync those we actually updated.
        if not self._dirty_experts:
            return

        for eid in sorted(self._dirty_experts):
            # Ensure both online and target experts are materialized on CPU so state_dict/load are safe
            self.expert_store.ensure_on_cpu([eid])
            self.target_store.ensure_on_cpu([eid])

            sd = {k: v.detach().cpu() for k, v in self.model.experts[eid].state_dict().items()}
            self.target_model.experts[eid].load_state_dict(sd, strict=True)

        self._dirty_experts.clear()

    def run_sleep(self, games_today: List[str]) -> Dict[str, object]:
        metrics: Dict[str, object] = {}
        updates = 0

        for game_id in games_today:
            if self.sleep_buffer.num_transitions(game_id) < self.cfg.seq_len:
                continue

            # Prefetch previously flagged experts for this game
            flagged = self.flagged_experts.get(game_id, [])
            if flagged:
                self.expert_store.prefetch_to_gpu(flagged)
                self.target_store.prefetch_to_gpu(flagged)

            self.expert_store.reset_stats()

            losses: List[float] = []
            used_counts: List[int] = []  # unique experts used per update

            for u in range(int(self.cfg.sleep_updates_per_game)):
                sample = self.sleep_buffer.sample_sequences(
                    game_id,
                    batch_size=self.cfg.batch_size,
                    seq_len=self.cfg.seq_len,
                    prioritize=True,
                    alpha=self.cfg.salience_alpha,
                )
                batch = self._to_torch_batch(sample)

                used_experts: Set[int] = set()

                # Pin experts used in this step so they can't be evicted mid-backward.
                if hasattr(self.expert_store, 'begin_step'):
                    self.expert_store.begin_step()
                try:
    
                    self.core_optim.zero_grad(set_to_none=True)
                    # Best-effort zero for experts currently on GPU; more precise would require tracking.
                    self.expert_store.zero_grad(self.expert_store.experts_on_gpu())
    
                    loss = self._loss_on_sequence_batch(batch, used_experts=used_experts)
                    loss.backward()
    
                    # Clip grads
                    torch.nn.utils.clip_grad_norm_(list(self.model.encoder.parameters()) + list(self.model.router.parameters()), 10.0)
                    for eid in used_experts:
                        torch.nn.utils.clip_grad_norm_(self.model.experts[eid].parameters(), 10.0)
    
                    self.core_optim.step()
                    self.expert_store.step(used_experts)
    
                    self._dirty_experts.update(used_experts)
                    if hasattr(self, '_dirty_since_night'):
                        self._dirty_since_night.update(used_experts)
                    used_counts.append(len(used_experts))
    
                    updates += 1
                    losses.append(float(loss.detach().cpu().item()))
    
                    if updates % self.cfg.target_update_interval == 0:
                        self._sync_target()
    
                finally:
                    if hasattr(self.expert_store, 'end_step'):
                        self.expert_store.end_step()

            # Fisher + snapshot after finishing this game in sleep
            fisher_filter = self._make_ewc_filter_fn(game_id)

            if self.cfg.protect_experts and flagged:
                # Ensure flagged experts are materialized on GPU before fisher init
                self.expert_store.ensure_on_gpu(flagged)

            fisher_kwargs: Dict[str, object] = {}
            try:
                sig = inspect.signature(estimate_fisher_diag)
                if 'begin_step_fn' in sig.parameters:
                    fisher_kwargs = {
                        'begin_step_fn': getattr(self.expert_store, 'begin_step', None),
                        'end_step_fn': getattr(self.expert_store, 'end_step', None),
                    }
                    fisher_kwargs = {k: v for k, v in fisher_kwargs.items() if v is not None}
            except Exception:
                fisher_kwargs = {}

            fisher = estimate_fisher_diag(
                self.model,
                batches=self._iter_fisher_batches(game_id, max_batches=self.cfg.fisher_batches),
                # Fisher should be estimated on the base learning objective (no EWC penalty).
                loss_fn=lambda m, b: self._loss_on_sequence_batch(b, used_experts=set(), include_ewc=False, include_moe_aux=False),
                filter_fn=fisher_filter,
                max_batches=self.cfg.fisher_batches,
                **fisher_kwargs,
            )
            snapshot = make_snapshot(self.model, name=game_id, fisher_diag=fisher, filter_fn=fisher_filter)
            self.ewc.add_task_snapshot(snapshot)

            # Router/expert diversity diagnostics during sleep updates
            if 'used_counts' in locals() and used_counts:
                metrics[f"sleep/{game_id}/unique_experts_mean"] = float(np.mean(used_counts))
                metrics[f"sleep/{game_id}/unique_experts_p95"] = float(np.percentile(used_counts, 95))
                metrics[f"sleep/{game_id}/unique_experts_max"] = float(np.max(used_counts))

            s = self.expert_store.reset_stats()
            metrics[f"sleep/{game_id}/hbm_hit_rate"] = s.hit_rate()
            metrics[f"sleep/{game_id}/stall_time_s"] = s.stall_time_s
            metrics[f"sleep/{game_id}/nvme_reads"] = s.nvme_reads
            metrics[f"sleep/{game_id}/nvme_writes"] = s.nvme_writes
            metrics[f"sleep/{game_id}/mean_loss"] = float(np.mean(losses)) if losses else 0.0
            metrics[f"sleep/{game_id}/updates"] = float(self.cfg.sleep_updates_per_game)

        metrics["sleep_total_updates"] = float(updates)
        return metrics

    # ------------------------ full cycle ------------------------

    def save_checkpoint(self, tag: str) -> Optional[Path]:
        if self.run_dir is None:
            return None

        # Persist any currently-loaded experts so Drive-backed run_dirs survive Colab restarts.
        if self.cfg.enable_nvme_tier:
            self.expert_store.flush_resident_to_disk()
            self.target_store.flush_resident_to_disk()

        ckpt_path = self.run_dir / "checkpoints" / f"ckpt_{tag}.pt"
        payload = {
            "global_step": int(self.global_step),
            "flagged_experts": dict(self.flagged_experts),
            "ewc_tasks": [
                {
                    "name": t.name,
                    "params_star": t.params_star,
                    "fisher_diag": t.fisher_diag,
                }
                for t in self.ewc.tasks
            ],
            "model_core": {
                "encoder": self.model.encoder.state_dict(),
                "router": self.model.router.state_dict(),
            },
            "target_core": {
                "encoder": self.target_model.encoder.state_dict(),
                "router": self.target_model.router.state_dict(),
            },
            "config": self.cfg.__dict__,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def _select_warm_set(self, games_today: List[str], day_summaries: Dict[str, DayStats]) -> List[int]:
        """Choose which experts to keep warm on CPU across days (DRAM tier).

        This is a simple heuristic that supports the design you described:
        - Router/backbone learns to page experts during the day.
        - At night we write back any dirty experts.
        - We keep a small warm DRAM set across days (capacity-limited).

        The policy here is intentionally lightweight:
        - Maintain a decayed retention score per expert.
        - Add today's per-game expert_scores into that retention score.
        - Always include base_experts.
        - Fill remaining DRAM capacity with top retention-score experts.
        """
        # Decay past retention so old tasks slowly cool off.
        retain_decay = float(getattr(self.cfg, 'retain_decay', getattr(self.cfg, 'warm_retain_decay', 0.9)))
        retain_decay = float(min(max(retain_decay, 0.0), 0.9999))
        if hasattr(self, '_retain_scores'):
            self._retain_scores *= retain_decay

        # Add today's expert usage/salience.
        for g in games_today:
            if g not in day_summaries:
                continue
            s = np.asarray(day_summaries[g].expert_scores, dtype=np.float32)
            if not hasattr(self, '_retain_scores') or s.shape != self._retain_scores.shape:
                continue
            denom = float(np.max(np.abs(s)) + 1e-6)
            self._retain_scores += (s / denom)

        cap = int(self.cfg.dram_expert_capacity)
        base = [int(x) for x in list(getattr(self, 'base_experts', []))]
        base = base[:cap]
        seen: Set[int] = set(base)

        remaining = max(0, cap - len(base))
        if remaining <= 0:
            return base

        order = np.argsort(-self._retain_scores).tolist() if hasattr(self, '_retain_scores') else list(range(self.cfg.num_experts))
        warm: List[int] = []
        for eid in order:
            eid = int(eid)
            if eid in seen:
                continue
            warm.append(eid)
            seen.add(eid)
            if len(warm) >= remaining:
                break

        return base + warm

    def run_one_day(self, day_index: int) -> Dict[str, object]:
        rng = random.Random(self.seed + day_index)
        games_today = rng.sample(self.games, k=min(self.cfg.games_per_day, len(self.games)))

        # Start-of-day: ensure the warm DRAM set is materialized, and prefetch the base set to GPU.
        if getattr(self, 'warm_experts', None) and self.cfg.enable_nvme_tier:
            self.expert_store.ensure_on_cpu(self.warm_experts)
            self.target_store.ensure_on_cpu(self.warm_experts)

        if getattr(self, 'base_experts', None):
            self.expert_store.prefetch_to_gpu(self.base_experts)
            self.target_store.prefetch_to_gpu(self.base_experts)


        day_summaries: Dict[str, DayStats] = {}
        for g in games_today:
            stats = self.run_day_for_game(g, steps=self.cfg.day_steps_per_game)
            day_summaries[g] = stats

            # Update flagged experts for this game based on today's usage
            topk = int(self.cfg.top_experts_per_game)
            top_ids = np.argsort(-stats.expert_scores)[:topk].tolist()
            self.flagged_experts[g] = [int(i) for i in top_ids]

        sleep_metrics = self.run_sleep(games_today)

        # Flush sleep buffer at end of sleep cycle
        self.sleep_buffer.flush()

        # End of night: write back dirty experts, retain a DRAM warm set across days, drop the rest to NVMe/meta.
        if self.cfg.enable_nvme_tier and hasattr(self.expert_store, 'reset_after_night'):
            dirty = set(getattr(self, '_dirty_since_night', set()))
            # Pick warm set for the next day (capacity-limited).
            warm_next = self._select_warm_set(games_today, day_summaries)
            self.warm_experts = list(warm_next)
            # Reset both stores. This will also write back dirty experts if provided.
            self.expert_store.reset_after_night(retain_cpu_ids=warm_next, writeback_ids=dirty, clear_hbm=True)
            self.target_store.reset_after_night(retain_cpu_ids=warm_next, writeback_ids=dirty, clear_hbm=True)
            self._dirty_since_night.clear()


        out: Dict[str, object] = {
            "day": int(day_index),
            "games_today": games_today,
            "episode_returns": {g: day_summaries[g].episode_returns for g in games_today},
            "episode_return_last": {
                g: (float(day_summaries[g].episode_returns[-1]) if day_summaries[g].episode_returns else 0.0)
                for g in games_today
            },
            "episode_return_mean": {
                g: (float(np.mean(day_summaries[g].episode_returns)) if day_summaries[g].episode_returns else 0.0)
                for g in games_today
            },
            "n_episodes": {g: int(len(day_summaries[g].episode_returns)) for g in games_today},
            "day_cache": {
                g: {
                    "hbm_hit_rate": day_summaries[g].store_stats.hit_rate(),
                    "stall_time_s": day_summaries[g].store_stats.stall_time_s,
                    "nvme_reads": day_summaries[g].store_stats.nvme_reads,
                    "nvme_writes": day_summaries[g].store_stats.nvme_writes,
                }
                for g in games_today
            },
            "flagged_experts": {g: self.flagged_experts[g] for g in games_today},
            "sleep": sleep_metrics,
            "ewc_tasks": len(self.ewc.tasks),
            "warm_dram_experts": list(getattr(self, 'warm_experts', [])),
        }

        # Save a lightweight checkpoint each day if run_dir is set
        self.save_checkpoint(tag=f"day{day_index:03d}")

        return out
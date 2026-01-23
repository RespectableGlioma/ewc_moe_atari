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
    unique_experts_used: int = 0


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

        # --- Results-regularized routing / expert spawning state ---
        # Pool of never-assigned expert ids (excluding the always-warm base experts).
        self._free_expert_ids: List[int] = [i for i in range(int(self.cfg.num_experts)) if i not in set(self.base_experts)]
        rng_spawn = random.Random(self.seed + 1337)
        rng_spawn.shuffle(self._free_expert_ids)

        # Per-game performance tracking used to decide whether to spawn a new expert.
        self._game_best_return: Dict[str, float] = {}

        # Optional: force newly spawned experts into the router's sparse top-k for a few sleep updates.
        self._force_experts: Dict[str, List[int]] = {}
        self._force_steps_remaining: Dict[str, int] = {}
        self._game_no_improve_days: Dict[str, int] = {}
        self._game_seen_days: Dict[str, int] = {}


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

        day_used: Set[int] = set()

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
            # Use only the *actually selected* experts (top-k) for scoring to avoid
            # "dense" bookkeeping when gating_probs is a full softmax over all experts.
            g_cpu = gating.detach().cpu()
            k = int(min(self.cfg.expert_top_k, g_cpu.numel()))
            if k > 0:
                topv, topi = torch.topk(g_cpu, k=k)
                expert_scores[topi.numpy()] += topv.numpy() * salience
                day_used.update([int(i) for i in topi.tolist()])

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
            unique_experts_used=int(len(day_used)),
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
    def _target_q_next(
        self,
        next_obs_t: torch.Tensor,
        hidden_t: torch.Tensor,
        *,
        allowed_experts: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            out_next = self.target_model(
                next_obs_t,
                hidden_t,
                expert_store=self.target_store,
                top_k=self.cfg.expert_top_k,
                allowed_experts=allowed_experts,
            )
        except TypeError:
            out_next = self.target_model(
                next_obs_t,
                hidden_t,
                expert_store=self.target_store,
                top_k=self.cfg.expert_top_k,
            )
        return out_next.q_values, out_next.hidden

    # ------------------------ learning (sleep) ------------------------

    def _loss_on_sequence_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        used_experts: Set[int],
        router_target_ids: Optional[Sequence[int]] = None,
        force_experts: Optional[Sequence[int]] = None,
        allowed_experts: Optional[Sequence[int]] = None,
        include_ewc: bool = True,
        include_results_aux: bool = True,
    ) -> torch.Tensor:
        """Compute DRQN-style DQN loss on a (B, T, ...) batch.

        This version adds an optional *results-regularized* router auxiliary term:

        - Provide `router_target_ids` (e.g., the current "assigned" experts for this game).
        - Penalize the router if it assigns low probability to those experts.
        - Weight this penalty more strongly when the TD-error is small (i.e., things are "working"),
          and weakly when TD-error is large (i.e., allow the router to deviate/explore).

        This is intentionally different from entropy/load-balance regularization: it is
        performance-shaped "stickiness", not generic diversity pressure.
        """
        obs = batch["obs"]            # (B, T, C, H, W)
        actions = batch["actions"]    # (B, T)
        rewards = batch["rewards"]    # (B, T)
        dones = batch["dones"]        # (B, T)
        next_obs = batch["next_obs"]  # (B, T, C, H, W)
        salience = batch["salience"]  # (B, T)

        B, T = actions.shape

        # Results-regularized router aux loss hyperparams (safe defaults if not in cfg).
        #
        # NOTE: This aux term is intentionally *results-shaped* ("stick with what works") rather
        # than generic entropy/load-balance pressure.
        results_coef = float(getattr(self.cfg, "router_results_coef", getattr(self.cfg, "router_results_weight", 0.10)))
        # Scale for TD-error -> stickiness weighting. Larger => router sticks more even when TD is high.
        td_scale = float(getattr(self.cfg, "router_results_td_scale", 1.0))
        td_scale = max(td_scale, 1e-6)

        aux_router = torch.zeros((), device=obs.device)
        aux_n = 0

        hidden = self.model.init_hidden(B, obs.device)
        total_loss = torch.zeros((), device=obs.device)

        # Precompute router target indices tensor once (if provided).
        target_idx = None
        if router_target_ids:
            # Dedup while preserving order
            uniq: List[int] = []
            seen: Set[int] = set()
            for i in router_target_ids:
                ii = int(i)
                if ii not in seen:
                    uniq.append(ii)
                    seen.add(ii)
            if uniq:
                target_idx = torch.as_tensor(uniq, device=obs.device, dtype=torch.long)

        for t in range(T):
            try:
                out = self.model(
                    obs[:, t],
                    hidden,
                    expert_store=self.expert_store,
                    top_k=self.cfg.expert_top_k,
                    record_used_experts=used_experts,
                    allowed_experts=allowed_experts,
                    force_experts=force_experts,
                )
            except TypeError:
                out = self.model(
                    obs[:, t],
                    hidden,
                    expert_store=self.expert_store,
                    top_k=self.cfg.expert_top_k,
                    record_used_experts=used_experts,
                )
            q = out.q_values
            hidden = out.hidden

            # Reset hidden where done
            done_t = dones[:, t].view(B, 1)
            hidden = hidden * (1.0 - done_t)

            a_t = actions[:, t].long().view(B, 1)
            q_sa = q.gather(1, a_t).squeeze(1)

            with torch.no_grad():
                q_next, _ = self._target_q_next(next_obs[:, t], hidden, allowed_experts=allowed_experts)
                max_next = q_next.max(dim=-1).values
                target = rewards[:, t] + (1.0 - dones[:, t]) * self.cfg.gamma * max_next

            td = q_sa - target
            huber = torch.where(td.abs() < 1.0, 0.5 * td.pow(2), td.abs() - 0.5)

            w = (salience[:, t].detach() + 1e-6) ** self.cfg.salience_alpha
            w = w / (w.mean() + 1e-6)

            total_loss = total_loss + (w * huber).mean()

            # Results-regularized router aux term: encourage probability mass on the assigned expert set,
            # with stronger pressure when TD-error is already low (i.e., stick with what works).
            if include_results_aux and results_coef > 0.0 and target_idx is not None:
                g = getattr(out, "gating_probs", None)
                if g is not None and g.numel() > 0:
                    eps = 1e-8
                    g = g / (g.sum(dim=-1, keepdim=True) + eps)

                    # Clamp indices defensively in case cfg.num_experts changed between runs.
                    idx = target_idx.clamp(0, g.shape[-1] - 1)

                    logp = (g.clamp_min(eps)).log()  # (B, E)
                    # Per-sample NLL of the "assigned" expert set (uniform target over that set).
                    nll = -logp.index_select(dim=1, index=idx).mean(dim=1)  # (B,)

                    # Results shaping: make the router *more sticky* when TD-error is already small.
                    # When TD-error is large, we reduce this pressure so the router can explore / switch.
                    td_abs = td.detach().abs()  # (B,)
                    sticky_w = torch.exp(-td_abs / td_scale).clamp(0.0, 1.0)  # (B,)
                    aux_router = aux_router + (sticky_w * nll).mean()

                    # Stickiness loss for the router (uniform target over the assigned expert set).
                    aux_n += 1

        # Add results-regularized router term (average over time steps).
        if include_results_aux and results_coef > 0.0 and aux_n > 0:
            total_loss = total_loss + results_coef * (aux_router / float(aux_n))

        if include_ewc:
            total_loss = total_loss + self.ewc.penalty(self.model)

        return total_loss

    def _estimate_fisher_diag_local(
        self,
        game_id: str,
        *,
        filter_fn,
        max_batches: int,
    ) -> Dict[str, torch.Tensor]:
        """Estimate a Fisher diagonal for the current model on a game's replay data.

        This is deliberately implemented here (rather than relying on the helper in ewc.py)
        because out-of-core expert paging + pinning can interact badly with generic Fisher
        utilities that temporarily freeze parameters or run under no_grad/inference_mode.

        Notes:
        - Uses include_ewc=False and include_moe_aux=False so Fisher reflects the *base* objective.
        - Pins experts via ExpertStore.begin_step()/end_step() so experts aren't evicted mid-backward.
        - Accumulates grad^2 on CPU to keep GPU memory stable.
        """
        max_batches = int(max(0, max_batches))
        if max_batches == 0:
            return {}

        fisher: Dict[str, torch.Tensor] = {}
        n = 0
        skipped = 0

        with torch.enable_grad():
            for batch in self._iter_fisher_batches(game_id, max_batches=max_batches):
                used_experts: Set[int] = set()

                if hasattr(self.expert_store, 'begin_step'):
                    self.expert_store.begin_step()
                try:
                    self.model.zero_grad(set_to_none=True)
                    # Best-effort zero for experts currently on GPU; avoids stale expert grads.
                    try:
                        self.expert_store.zero_grad(self.expert_store.experts_on_gpu())
                    except Exception:
                        pass

                    loss = self._loss_on_sequence_batch(
                        batch,
                        used_experts=used_experts,
                        include_ewc=False,
                        include_results_aux=False,
                    )
                    if not getattr(loss, 'requires_grad', False):
                        skipped += 1
                        continue

                    loss.backward()

                    for name, p in self.model.named_parameters():
                        if filter_fn is not None and not filter_fn(name):
                            continue
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if getattr(g, 'is_meta', False):
                            continue
                        g2 = (g.float() ** 2).cpu()
                        if name in fisher:
                            fisher[name] += g2
                        else:
                            fisher[name] = g2

                    n += 1
                finally:
                    if hasattr(self.expert_store, 'end_step'):
                        self.expert_store.end_step()

        if n > 0:
            for k in list(fisher.keys()):
                fisher[k] /= float(n)
        else:
            if skipped > 0:
                print(
                    f"[DayNightTrainer] WARNING: fisher for {game_id} used 0/{max_batches} batches "
                    f"(loss.requires_grad was False). Returning empty fisher."
                )
        return fisher

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

                # Optional: restrict router to a per-game allow-list (base + flagged).
                allowed_experts = None
                if bool(getattr(self.cfg, "restrict_router_to_flagged", False)) and flagged:
                    # keep order, unique
                    allowed_experts = list(dict.fromkeys(list(self.base_experts) + list(flagged)))

                # Optional: temporarily force a newly spawned expert into the sparse top-k so it gets trained.
                force_experts = None
                _force_left = int(self._force_steps_remaining.get(game_id, 0))
                if _force_left > 0:
                    force_experts = self._force_experts.get(game_id, None)

                # Pin experts used in this step so they can't be evicted mid-backward.
                if hasattr(self.expert_store, 'begin_step'):
                    self.expert_store.begin_step()
                try:
    
                    self.core_optim.zero_grad(set_to_none=True)
                    # Best-effort zero for experts currently on GPU; more precise would require tracking.
                    self.expert_store.zero_grad(self.expert_store.experts_on_gpu())
    
                    loss = self._loss_on_sequence_batch(
                        batch,
                        used_experts=used_experts,
                        router_target_ids=flagged,
                        force_experts=force_experts,
                        allowed_experts=allowed_experts,
                    )
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
                    # Decrement force counter (if any) after each update
                    if _force_left > 0:
                        _force_left -= 1
                        if _force_left <= 0:
                            self._force_steps_remaining.pop(game_id, None)
                            self._force_experts.pop(game_id, None)
                        else:
                            self._force_steps_remaining[game_id] = _force_left
                    losses.append(float(loss.detach().cpu().item()))
    
                    if updates % self.cfg.target_update_interval == 0:
                        self._sync_target()
    
                finally:
                    if hasattr(self.expert_store, 'end_step'):
                        self.expert_store.end_step()
            # Fisher + snapshot after finishing this game in sleep
            fisher_filter = self._make_ewc_filter_fn(game_id)

            if self.cfg.protect_experts and flagged:
                # Ensure flagged experts are materialized on GPU before fisher init.
                # (Forward will page others as needed, but this avoids edge-cases where
                # the filter_fn includes experts that never get touched during fisher.)
                self.expert_store.ensure_on_gpu(flagged)

            fisher = self._estimate_fisher_diag_local(
                game_id,
                filter_fn=fisher_filter,
                max_batches=int(self.cfg.fisher_batches),
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
            metrics[f"sleep/{game_id}/bytes_nvme_read"] = float(getattr(s, "bytes_nvme_read", 0.0))
            metrics[f"sleep/{game_id}/bytes_nvme_write"] = float(getattr(s, "bytes_nvme_write", 0.0))
            metrics[f"sleep/{game_id}/bytes_h2d"] = float(getattr(s, "bytes_h2d", 0.0))
            metrics[f"sleep/{game_id}/bytes_d2h"] = float(getattr(s, "bytes_d2h", 0.0))
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


    # ---------------------------------------------------------------------
    # Results-regularized expert assignment + spawning helpers
    # ---------------------------------------------------------------------

    def _alloc_new_expert_id(self) -> Optional[int]:
        """Allocate an expert id for "spawn a new expert" behavior.

        Prefer never-assigned ids from self._free_expert_ids. If that pool is empty,
        fall back to recycling the *lowest-retention* expert that is not:
          - in the always-warm base set
          - currently in the warm DRAM set
          - currently assigned to any active game's flagged expert set
        """
        # Fast path: never-assigned pool.
        if getattr(self, "_free_expert_ids", None):
            if len(self._free_expert_ids) > 0:
                return int(self._free_expert_ids.pop())

        # Slow path: recycle a cold/low-value expert id.
        base = set(getattr(self, "base_experts", []))
        warm = set(getattr(self, "warm_experts", []))
        assigned: Set[int] = set()
        for v in getattr(self, "flagged_experts", {}).values():
            for eid in v:
                assigned.add(int(eid))

        candidates = [i for i in range(int(self.cfg.num_experts)) if i not in base and i not in warm and i not in assigned]
        if not candidates:
            return None

        scores = getattr(self, "_retain_scores", None)
        if scores is None:
            return int(random.choice(candidates))

        # Pick the minimum retention score.
        best_i = min(candidates, key=lambda i: float(scores[int(i)]))
        return int(best_i)

    
    def _init_expert_set_for_game(self, game_id: str) -> List[int]:
        """Initialize a per-game expert set the first time we see a game.

        Design intent:
        - Do **not** pre-partition experts by game/task.
        - Let the shared router/backbone predict which experts to call from context.
        - After the day phase, we *consolidate* a game's expert set from the experts that
          were actually used/credited during that day.

        Therefore the default initial assignment is empty.
        """
        return []



    def _consolidate_flagged_experts_from_day(
        self,
        game_id: str,
        stats: DayStats,
        top_ids: Sequence[int],
    ) -> List[int]:
        """Update a game's `flagged_experts` after the day rollouts.

        We want *learned* routing rather than fixed partitions. So we:
        - take today's top-used experts (salience-weighted credit)
        - optionally blend with last day's flagged set to reduce churn
        - store the resulting set as `flagged_experts[game_id]` which is then:
            * prefetched during future days
            * used as the router's "results-regularized" target during sleep
            * used by EWC to decide which expert params to protect
        """
        topk = int(self.cfg.top_experts_per_game)
        prev = [int(x) for x in self.flagged_experts.get(game_id, [])]
        top = [int(x) for x in list(top_ids)]

        # Candidate pool = prev ∪ top (preserve order).
        candidates: List[int] = []
        seen: Set[int] = set()
        for eid in prev + top:
            eid = int(eid)
            if eid not in seen:
                candidates.append(eid)
                seen.add(eid)

        if not candidates or topk <= 0:
            self.flagged_experts[game_id] = []
            return []

        scores = np.asarray(stats.expert_scores, dtype=np.float32)

        # Stickiness bonus (scaled to score magnitude so it's roughly dimensionless).
        stick = float(getattr(self.cfg, "router_assignment_stickiness", getattr(self.cfg, "flagged_stickiness", 0.05)))
        stick = float(max(stick, 0.0))
        bonus = stick * float(np.max(np.abs(scores)) + 1e-6)

        prev_set = set(prev)

        def score_of(eid: int) -> float:
            s = float(scores[eid]) if 0 <= eid < int(scores.shape[0]) else 0.0
            if eid in prev_set:
                s += bonus
            return s

        ranked = sorted(candidates, key=score_of, reverse=True)
        out = [int(e) for e in ranked[:topk]]
        self.flagged_experts[game_id] = out
        return out


    def _maybe_spawn_new_experts(
        self,
        *,
        day_index: int,
        games_today: List[str],
        day_summaries: Dict[str, "DayStats"],
    ) -> Dict[str, int]:
        """Decide which games should get a freshly-initialized expert before sleep.

        Heuristic (tunable via cfg):
        - Track each game's mean episode return across days.
        - If a game's mean return fails to improve for `spawn_patience_days` consecutive days
          AND the mean return is <= `spawn_min_return`, then it becomes eligible.
        - Spawn at most `max_spawns_per_day` experts (pick worst-performing games).

        Returns: mapping game_id -> spawned_expert_id
        """
        max_spawns = int(getattr(self.cfg, "max_spawns_per_day", getattr(self.cfg, "spawn_max_per_day", 0)))
        if max_spawns <= 0:
            return {}

        patience = int(getattr(self.cfg, "spawn_patience_days", getattr(self.cfg, "spawn_patience", 2)))
        patience = max(patience, 1)

        improve_eps = float(getattr(self.cfg, "spawn_improve_eps", 0.0))
        min_return = float(getattr(self.cfg, "spawn_min_return", 0.0))
        retain_bonus = float(getattr(self.cfg, "spawn_retain_bonus", 5.0))

        # Update trackers and collect eligible games.
        eligible: List[Tuple[float, str]] = []
        for g in games_today:
            stats = day_summaries[g]
            mean_ret = float(np.mean(stats.episode_returns)) if stats.episode_returns else 0.0

            self._game_seen_days[g] = int(self._game_seen_days.get(g, 0) + 1)

            best = float(self._game_best_return.get(g, -1e9))
            no_imp = int(self._game_no_improve_days.get(g, 0))

            if mean_ret > best + improve_eps:
                best = mean_ret
                no_imp = 0
            else:
                no_imp += 1

            self._game_best_return[g] = best
            self._game_no_improve_days[g] = no_imp

            if no_imp >= patience and mean_ret <= min_return:
                eligible.append((mean_ret, g))

        if not eligible:
            return {}

        eligible.sort(key=lambda x: x[0])  # worst first

        spawned: Dict[str, int] = {}
        topk = int(self.cfg.top_experts_per_game)

        for _, g in eligible[:max_spawns]:
            new_id = self._alloc_new_expert_id()
            if new_id is None:
                break

            # Re-init the expert weights (online + target copies).
            seed = (self.seed + 1000003 * int(day_index) + (abs(hash(g)) % 1000003)) % (2**31 - 1)
            if hasattr(self.expert_store, "reinit_expert"):
                self.expert_store.reinit_expert(new_id, seed=seed, save_to_disk=True, move_to="cpu")
            if hasattr(self.target_store, "reinit_expert"):
                self.target_store.reinit_expert(new_id, seed=seed, save_to_disk=True, move_to="cpu")

            # Force the newly spawned expert into the router's sparse top-k for a short period so it gets trained.
            force_steps = int(getattr(self.cfg, "spawn_force_sleep_steps", 50))
            if force_steps > 0:
                self._force_experts[g] = [int(new_id)]
                self._force_steps_remaining[g] = force_steps


            # Replace the lowest-scoring expert in this game's current set.
            scores = day_summaries[g].expert_scores
            cur = list(self.flagged_experts.get(g, []))
            if not cur:
                # Fall back to today's top-used experts (or at least something) before adding the new expert.
                top_ids = np.argsort(-scores)[:max(1, topk)].tolist()
                cur = [int(i) for i in top_ids]
            if len(cur) < topk:
                cur.append(int(new_id))
            else:
                # Choose the assigned expert with the lowest credit/usage today.
                replace_i = int(np.argmin([float(scores[int(e)]) for e in cur])) if len(cur) > 0 else 0
                cur[replace_i] = int(new_id)

            # Dedup and truncate.
            uniq: List[int] = []
            seen: Set[int] = set()
            for e in cur:
                e = int(e)
                if e not in seen:
                    uniq.append(e)
                    seen.add(e)
            self.flagged_experts[g] = uniq[:topk]

            # Give the new expert a retention "boost" so it's more likely to stay warm across the next day.
            if hasattr(self, "_retain_scores") and self._retain_scores is not None:
                try:
                    self._retain_scores[int(new_id)] += float(retain_bonus)
                except Exception:
                    pass

            # Reset staleness so we don't instantly spawn again tomorrow.
            self._game_no_improve_days[g] = 0

            spawned[g] = int(new_id)

        return spawned


    def run_one_day(self, day_index: int) -> Dict[str, object]:
            rng = random.Random(self.seed + day_index)
            games_today = rng.sample(self.games, k=min(self.cfg.games_per_day, len(self.games)))

            # Ensure every game we might touch today already has an assigned expert set.
            for g in games_today:
                if g not in self.flagged_experts:
                    self.flagged_experts[g] = self._init_expert_set_for_game(g)

            # Start-of-day: ensure the warm DRAM set is materialized, and prefetch the base set to GPU.
            if getattr(self, 'warm_experts', None) and self.cfg.enable_nvme_tier:
                self.expert_store.ensure_on_cpu(self.warm_experts)
                self.target_store.ensure_on_cpu(self.warm_experts)

            if getattr(self, 'base_experts', None):
                self.expert_store.prefetch_to_gpu(self.base_experts)
                self.target_store.prefetch_to_gpu(self.base_experts)

            day_summaries: Dict[str, DayStats] = {}
            day_top_used: Dict[str, List[int]] = {}

            for g in games_today:
                stats = self.run_day_for_game(g, steps=self.cfg.day_steps_per_game)
                day_summaries[g] = stats

                # For diagnostics only: which experts were most credited/used today?
                topk = int(self.cfg.top_experts_per_game)
                top_ids = np.argsort(-stats.expert_scores)[:topk].tolist()
                day_top_used[g] = [int(i) for i in top_ids]

                # Consolidate a per-game expert set from today's *actual* usage.
                # This avoids hard-coding a disjoint expert partition by game.
                self._consolidate_flagged_experts_from_day(g, stats, day_top_used[g])


            # Before sleep: decide whether any games should "spawn" a fresh expert.
            spawned = self._maybe_spawn_new_experts(day_index=day_index, games_today=games_today, day_summaries=day_summaries)

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
                        "bytes_nvme_read": float(getattr(day_summaries[g].store_stats, "bytes_nvme_read", 0.0)),
                        "bytes_nvme_write": float(getattr(day_summaries[g].store_stats, "bytes_nvme_write", 0.0)),
                        "bytes_h2d": float(getattr(day_summaries[g].store_stats, "bytes_h2d", 0.0)),
                        "bytes_d2h": float(getattr(day_summaries[g].store_stats, "bytes_d2h", 0.0)),
                        "unique_experts_used": int(getattr(day_summaries[g], "unique_experts_used", 0)),
                    }
                    for g in games_today
                },
                "flagged_experts": {g: self.flagged_experts[g] for g in games_today},
                "day_top_used_experts": day_top_used,
                "spawned_experts": spawned,
                "sleep": sleep_metrics,
                "ewc_tasks": len(self.ewc.tasks),
                "warm_dram_experts": list(getattr(self, 'warm_experts', [])),
            }

            # Save a lightweight checkpoint each day if run_dir is set
            self.save_checkpoint(tag=f"day{day_index:03d}")

            return out

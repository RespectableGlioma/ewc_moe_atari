#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ooc_moe.atari import AtariBatchSource, AtariEnvConfig
from ooc_moe.moe import moe_forward_top1, route_fixed_by_env
from ooc_moe.synthetic import EnvEpisodeSampler, EnvMarkovSampler, SyntheticEnvTeacher
from ooc_moe.tiered_store import LatencySim, TieredExpertStore
from ooc_moe.utils import Stats


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


def _try_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


class FixedObsFeaturizer(nn.Module):
    """A *non-trainable* featurizer used to create a stable teacher target.

    We pool (avg) the stacked frames and then apply a fixed random projection.
    This avoids the trivial-collapse issue that happens if the teacher depends on
    a learnable student encoder.

    Input:  obs float32 in [0,1], shape (B, C, H, W)
    Output: features float32, shape (B, d_model)
    """

    def __init__(self, *, c: int, h: int, w: int, d_model: int, pool: int, seed: int, device: torch.device):
        super().__init__()
        if pool <= 0:
            raise ValueError("pool must be > 0")
        self.pool = int(pool)
        h2, w2 = h // pool, w // pool
        if h2 <= 0 or w2 <= 0:
            raise ValueError(f"pool={pool} too large for HxW={h}x{w}")
        in_dim = int(c * h2 * w2)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        w_proj = torch.randn(d_model, in_dim, generator=g, dtype=torch.float32) / (in_dim**0.5)
        self.register_buffer("w_proj", w_proj.to(device=device), persistent=False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B,C,H,W)
        x = F.avg_pool2d(obs, kernel_size=self.pool, stride=self.pool)
        x = x.flatten(1)
        return F.linear(x, self.w_proj)


class ConvStudentEncoder(nn.Module):
    """A small trainable encoder from Atari frames -> d_model."""

    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # For 84x84 input, this is 7x7 spatial.
        self.fc = nn.Linear(64 * 7 * 7, d_model)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.conv(obs)
        h = h.flatten(1)
        return self.fc(h)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Out-of-core MoE training prototype on real Atari observations (ALE / Gymnasium).",
    )

    # --- Expert-cache / MoE knobs (same as train.py) ---
    ap.add_argument("--n_experts", type=int, default=512)
    ap.add_argument("--n_envs", type=int, default=512, help="Number of *logical* env IDs (routes/teachers).")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--gpu_slots", type=int, default=16)
    ap.add_argument("--cpu_cache", type=int, default=64)
    ap.add_argument("--disk_root", type=str, default="")
    ap.add_argument("--reset_disk", action="store_true")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])

    ap.add_argument("--sampler", type=str, default="episode", choices=["episode", "markov"])
    ap.add_argument("--episode_len", type=int, default=30)
    ap.add_argument("--p_stay", type=float, default=0.9)
    ap.add_argument("--envs_per_batch", type=int, default=1)
    ap.add_argument("--sort_by_expert", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument("--optim", type=str, default="adamw", choices=["sgd", "sgd_momentum", "adamw"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_eps", type=float, default=1e-8)

    ap.add_argument("--writeback_policy", type=str, default="evict", choices=["evict", "periodic", "writethrough"])
    ap.add_argument("--writeback_every", type=int, default=0)
    ap.add_argument("--io_workers", type=int, default=2)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--prefetch_gpu", action="store_true")
    ap.add_argument("--prefetch_gpu_max", type=int, default=0)

    ap.add_argument("--sim_h2d_gbps", type=float, default=0.0)
    ap.add_argument("--sim_d2h_gbps", type=float, default=0.0)
    ap.add_argument("--sim_disk_read_gbps", type=float, default=0.0)
    ap.add_argument("--sim_disk_write_gbps", type=float, default=0.0)
    ap.add_argument("--sim_extra_ms_per_io", type=float, default=0.0)

    ap.add_argument("--no_tqdm", action="store_true")

    # --- Atari knobs ---
    ap.add_argument(
        "--games",
        type=str,
        default="Pong,Breakout,SpaceInvaders,Seaquest,Qbert",
        help="Comma-separated Atari game names or env IDs (e.g. ALE/Pong-v5).",
    )
    ap.add_argument("--env_backend", type=str, default="ale_vec", choices=["ale_vec", "wrappers"])
    ap.add_argument("--vec_envs_per_game", type=int, default=8)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--stack_size", type=int, default=4)
    ap.add_argument("--screen_size", type=int, default=84)
    ap.add_argument("--no_grayscale", action="store_true")
    ap.add_argument("--repeat_action_probability", type=float, default=0.0)
    ap.add_argument("--use_fire_reset", action="store_true")
    ap.add_argument("--reward_clipping", action="store_true")
    ap.add_argument("--episodic_life", action="store_true")

    # --- Teacher / encoder knobs ---
    ap.add_argument("--teacher_noise", type=float, default=0.01)
    ap.add_argument("--teacher_pool", type=int, default=4, help="AvgPool kernel/stride for teacher featurizer.")
    ap.add_argument("--encoder", type=str, default="none", choices=["none", "conv"])
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--encoder_wd", type=float, default=0.0)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    compute_dtype = parse_dtype(args.dtype)
    disk_root = Path(args.disk_root) if args.disk_root else None
    if disk_root is not None:
        if args.reset_disk and disk_root.exists():
            shutil.rmtree(disk_root, ignore_errors=True)
        disk_root.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    lat = LatencySim(
        h2d_gbps=args.sim_h2d_gbps,
        d2h_gbps=args.sim_d2h_gbps,
        disk_read_gbps=args.sim_disk_read_gbps,
        disk_write_gbps=args.sim_disk_write_gbps,
        extra_ms_per_io=args.sim_extra_ms_per_io,
    )

    store = TieredExpertStore(
        n_experts=args.n_experts,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        gpu_slots=args.gpu_slots,
        cpu_cache_capacity=args.cpu_cache,
        device=device,
        compute_dtype=compute_dtype if device.type == "cuda" else torch.float32,
        activation=args.activation,
        disk_root=disk_root,
        pin_cpu=True,
        optim=args.optim,
        momentum=args.momentum,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        writeback_policy=args.writeback_policy,
        writeback_every=args.writeback_every,
        io_workers=args.io_workers,
        latency_sim=lat,
        stats=stats,
        seed=args.seed,
    )

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    atari_cfg = AtariEnvConfig(
        backend=args.env_backend,
        num_envs=args.vec_envs_per_game,
        frameskip=args.frameskip,
        stack_size=args.stack_size,
        noop_max=30,
        img_height=args.screen_size,
        img_width=args.screen_size,
        grayscale=not args.no_grayscale,
        repeat_action_probability=args.repeat_action_probability,
        use_fire_reset=args.use_fire_reset,
        reward_clipping=args.reward_clipping,
        episodic_life=args.episodic_life,
    )
    src = AtariBatchSource(games, cfg=atari_cfg, seed=args.seed + 123)

    # Determine observation shape so we can set up the featurizers.
    probe = src.sample(env_id=0, n=1)
    # Expected: (1, stack, H, W) for grayscale.
    if probe.ndim == 4:
        _, c, h, w = probe.shape
    elif probe.ndim == 5:
        # e.g. (1, stack, H, W, 3)
        _, c, h, w, _rgb = probe.shape
    else:
        raise RuntimeError(f"Unexpected Atari observation shape: {probe.shape}")

    teacher_feat = FixedObsFeaturizer(
        c=int(c),
        h=int(h),
        w=int(w),
        d_model=int(args.d_model),
        pool=int(args.teacher_pool),
        seed=int(args.seed + 999),
        device=device,
    )

    encoder: Optional[nn.Module]
    enc_optim: Optional[torch.optim.Optimizer]
    if args.encoder == "conv":
        encoder = ConvStudentEncoder(in_channels=int(c), d_model=int(args.d_model)).to(device=device)
        enc_optim = torch.optim.AdamW(encoder.parameters(), lr=float(args.encoder_lr), weight_decay=float(args.encoder_wd))
    else:
        encoder = None
        enc_optim = None

    # Teacher matrices live on CPU by default; optionally moved to GPU for speed.
    # We keep noise separate so it's applied exactly once.
    teacher = SyntheticEnvTeacher(args.n_envs, args.d_model, seed=args.seed, noise_std=0.0)
    teacher_A = teacher.A
    if device.type == "cuda":
        teacher_A = teacher_A.to(device=device, dtype=torch.float32)
    if args.sampler == "episode":
        sampler = EnvEpisodeSampler(args.n_envs, episode_len=args.episode_len, seed=args.seed + 1)
    else:
        sampler = EnvMarkovSampler(args.n_envs, p_stay=args.p_stay, seed=args.seed + 1)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 2)

    tqdm = None if args.no_tqdm else _try_tqdm()
    iterator = range(1, args.steps + 1) if tqdm is None else tqdm(range(1, args.steps + 1))

    def make_env_ids() -> torch.Tensor:
        envs = [sampler.next() for _ in range(args.envs_per_batch)]
        ids = torch.tensor(
            [envs[int(torch.randint(0, len(envs), (1,), generator=g).item())] for _ in range(args.batch_size)],
            device=device,
            dtype=torch.long,
        )
        return ids

    def sample_obs_for_env_ids(env_ids: torch.Tensor) -> torch.Tensor:
        # env_ids is on device; move to CPU for indexing / grouping.
        env_cpu = env_ids.detach().to("cpu")
        obs = None
        for eid in torch.unique(env_cpu).tolist():
            idx = (env_cpu == int(eid)).nonzero(as_tuple=False).squeeze(-1).numpy()
            chunk = src.sample(int(eid), int(idx.shape[0]))
            chunk_t = torch.from_numpy(chunk)
            if obs is None:
                obs = torch.empty((env_cpu.shape[0],) + tuple(chunk_t.shape[1:]), dtype=chunk_t.dtype)
            obs[idx] = chunk_t
        assert obs is not None
        # Convert to (B,C,H,W) float in [0,1]
        if obs.ndim == 5:
            # (B, stack, H, W, 3) -> drop RGB by averaging (keeps teacher stable).
            obs = obs.float().mean(dim=-1)
        obs = obs.to(device=device, dtype=torch.float32) / 255.0
        return obs

    # Lookahead buffers for prefetch
    cur_env_ids = make_env_ids()
    cur_expert_ids = route_fixed_by_env(cur_env_ids, n_experts=args.n_experts)

    next_env_ids = make_env_ids()
    next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
    if args.prefetch:
        store.prefetch(torch.unique(next_expert_ids).tolist())
    if args.prefetch_gpu:
        store.prefetch_to_gpu(torch.unique(next_expert_ids).tolist(), max_items=args.prefetch_gpu_max)

    t0_all = time.perf_counter()
    frames = 0

    for step in iterator:
        n_unique = int(torch.unique(cur_expert_ids).numel())
        if n_unique > args.gpu_slots:
            raise RuntimeError(
                f"Batch uses {n_unique} unique experts but gpu_slots={args.gpu_slots}. "
                f"Increase --gpu_slots or reduce --envs_per_batch / batch_size."
            )

        # Drain any completed prefetch work and ensure current experts are resident.
        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)
        for eid in torch.unique(cur_expert_ids).tolist():
            store.ensure_on_gpu(int(eid))

        # Launch GPU prefetch for next-step experts into remaining free slots.
        if args.prefetch_gpu:
            store.prefetch_to_gpu(torch.unique(next_expert_ids).tolist(), max_items=args.prefetch_gpu_max)

        # --- Collect observations (CPU) ---
        obs = sample_obs_for_env_ids(cur_env_ids)
        frames += int(obs.shape[0])

        # Teacher features are fixed.
        x_teacher = teacher_feat(obs)
        # Compute y = A_env x_teacher (same teacher form as SyntheticEnvTeacher, but using observation-derived x).
        if teacher_A.device == cur_env_ids.device:
            A_batch = teacher_A.index_select(0, cur_env_ids)
        else:
            A_batch = teacher_A.index_select(0, cur_env_ids.to("cpu")).to(device=device, dtype=torch.float32)
        y_true = torch.einsum("bd,bde->be", x_teacher, A_batch)
        if args.teacher_noise > 0:
            y_true = y_true + torch.randn_like(y_true) * float(args.teacher_noise)

        if encoder is None:
            x_in = x_teacher
        else:
            x_in = encoder(obs)

        if device.type == "cuda":
            x_in = x_in.to(dtype=compute_dtype)
            y_true = y_true.to(dtype=compute_dtype)

        if enc_optim is not None:
            enc_optim.zero_grad(set_to_none=True)

        y_pred = moe_forward_top1(x_in, cur_expert_ids, store, sort_by_expert=args.sort_by_expert)
        loss = torch.mean((y_pred - y_true) ** 2)
        loss.backward()

        # Update experts out-of-core.
        for eid in torch.unique(cur_expert_ids).tolist():
            store.step_expert(int(eid), lr=args.lr, weight_decay=args.weight_decay)

        # Update encoder in-core (optional).
        if enc_optim is not None:
            enc_optim.step()

        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)

        if device.type == "cuda" and (step % args.log_every == 0 or step == 1):
            torch.cuda.synchronize()

        if step % args.log_every == 0 or step == 1:
            loss_val = float(loss.detach().cpu().item())
            # Show the current game id for intuition.
            cur_e0 = int(cur_env_ids[0].detach().cpu().item())
            game_idx = src.env_id_to_game_index(cur_e0)
            game_name = src.env_ids[game_idx]
            msg = (
                f"step={step:5d} loss={loss_val:.6f} unique_experts={n_unique} "
                f"gpu_cache={len(store.gpu_map)} cpu_cache={len(store.cpu_cache)} "
                f"env0={cur_e0} game={game_name}"
            )
            if tqdm is None:
                print(msg)
            else:
                iterator.set_description(msg)

        # Shift lookahead and schedule prefetch.
        cur_env_ids, cur_expert_ids = next_env_ids, next_expert_ids
        next_env_ids = make_env_ids()
        next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
        if args.prefetch:
            store.prefetch(torch.unique(next_expert_ids).tolist())

    t_all = time.perf_counter() - t0_all
    if tqdm is not None:
        iterator.close()
    print(f"done in {t_all:.2f}s, approx frames={frames} ({frames / max(1e-9, t_all):.1f} frames/s)")

    src.close()
    store.flush_all()
    print()
    print(stats.summary())


if __name__ == "__main__":
    main()

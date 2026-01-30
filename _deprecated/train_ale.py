#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
import json
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
from ooc_moe.persist import JsonlLogger, atomic_torch_save, atomic_write_json, is_colab, mount_gdrive


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


class FixedGameTeacher:
    """A fixed per-game teacher with a fixed gating rule (used to supervise routing).

    For each game g, we create:
      - gate[g]: [K, d_model] random vectors -> bucket = argmax(x @ gate[g]^T)
      - A[g]: [K, d_model, d_model] random linear maps applied to x

    This creates a non-trivial "within-game regimes" target so a learned router has
    something to do beyond simply identifying the game.
    """

    def __init__(self, n_games: int, k: int, d_model: int, *, seed: int, device: torch.device):
        self.n_games = int(n_games)
        self.k = int(k)
        self.d_model = int(d_model)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        gate = torch.randn(self.n_games, self.k, self.d_model, generator=g, dtype=torch.float32) / (self.d_model**0.5)
        A = torch.randn(
            self.n_games,
            self.k,
            self.d_model,
            self.d_model,
            generator=g,
            dtype=torch.float32,
        ) / (self.d_model**0.5)

        self.gate = gate.to(device=device)
        self.A = A.to(device=device)

    @torch.no_grad()
    def labels_and_targets(self, x: torch.Tensor, game_ids: torch.Tensor, *, noise_std: float = 0.0):
        """Return (labels, y) given x=[B,d_model] and game_ids=[B]."""
        B, d = x.shape
        labels = torch.empty((B,), device=x.device, dtype=torch.long)
        y = torch.empty((B, d), device=x.device, dtype=x.dtype)

        for gi in torch.unique(game_ids).tolist():
            idx = (game_ids == int(gi)).nonzero(as_tuple=False).squeeze(-1)
            xg = x.index_select(0, idx)
            gate_g = self.gate[int(gi)]  # [K,d]
            logits = xg @ gate_g.t()
            lab = torch.argmax(logits, dim=-1)
            labels.index_copy_(0, idx, lab)

            A_g = self.A[int(gi)]  # [K,d,d]
            Ag_lab = A_g.index_select(0, lab)  # [n_i,d,d]
            yg = torch.einsum("nd,ndm->nm", xg, Ag_lab)
            y.index_copy_(0, idx, yg)

        if noise_std > 0:
            y = y + torch.randn_like(y) * float(noise_std)
        return labels, y


class PerGameRouter(nn.Module):
    """A simple per-game router: one linear head per game, mapping x -> logits over K experts."""

    def __init__(self, n_games: int, d_model: int, k: int):
        super().__init__()
        self.n_games = int(n_games)
        self.k = int(k)
        self.heads = nn.ModuleList([nn.Linear(int(d_model), int(k)) for _ in range(int(n_games))])

    def forward(self, x: torch.Tensor, game_id: int) -> torch.Tensor:
        return self.heads[int(game_id)](x)


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

    # --- Routing / router learning (semi-fixed routing per game) ---
    ap.add_argument(
        "--routing_mode",
        type=str,
        default="fixed_env",
        choices=["fixed_env", "game_router"],
        help="fixed_env: expert = env_id % n_experts (baseline). "
        "game_router: fixed expert sets per game + learned selector within that set.",
    )
    ap.add_argument("--experts_per_game", type=int, default=8, help="Only used for routing_mode=game_router.")
    ap.add_argument("--route_sample", action="store_true", help="Sample from router softmax instead of argmax.")
    ap.add_argument("--route_temperature", type=float, default=1.0, help="Softmax temperature for routing.")
    ap.add_argument("--router_lr", type=float, default=1e-3)
    ap.add_argument("--router_wd", type=float, default=0.0)
    ap.add_argument("--router_loss_weight", type=float, default=0.1, help="Weight on router cross-entropy loss.")
    ap.add_argument(
        "--route_miss_penalty",
        type=float,
        default=0.0,
        help="If >0, subtract miss_cost(expert) * route_miss_penalty from router logits.",
    )
    ap.add_argument("--miss_cost_gpu", type=float, default=0.0)
    ap.add_argument("--miss_cost_cpu", type=float, default=0.2)
    ap.add_argument("--miss_cost_disk", type=float, default=1.0)
    ap.add_argument("--no_pipeline_env", action="store_true", help="Disable ALE send/recv pipelining (debug).")

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

    # --- Persistence (useful in Colab) ---
    ap.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="If set, write metrics + checkpoints here (e.g. /content/drive/MyDrive/Colab_Notebooks/ooc_runs/ale_run1).",
    )
    ap.add_argument(
        "--mount_drive",
        action="store_true",
        help="If running in Colab, attempt to mount Google Drive at /content/drive before using --run_dir.",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help="If >0, flush dirty expert state and save a checkpoint every N steps.",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from --run_dir/checkpoint_last.pt")

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

    if args.mount_drive:
        # Best-effort; in Colab you normally mount once in a notebook cell.
        try:
            mount_gdrive("/content/drive")
        except Exception as e:
            print(f"[warn] drive mount failed: {e}")

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.resume and run_dir is None:
        raise ValueError("--resume requires --run_dir")
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    compute_dtype = parse_dtype(args.dtype)
    # If run_dir is provided and disk_root isn't, default the disk backing store to run_dir/disk
    if args.disk_root:
        disk_root = Path(args.disk_root)
    elif run_dir is not None:
        disk_root = run_dir / "disk"
    else:
        disk_root = None
    if disk_root is not None:
        if args.resume:
            # Don't destroy state when resuming.
            pass
        else:
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
    probe = src.sample_from_buffer(env_id=0, n=1)
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



    router: Optional[PerGameRouter] = None
    router_optim: Optional[torch.optim.Optimizer] = None
    if args.routing_mode == "game_router":
        n_games = len(src.envs)
        k = int(args.experts_per_game)
        router = PerGameRouter(n_games=n_games, d_model=int(args.d_model), k=k).to(device=device)
        router_optim = torch.optim.AdamW(
            router.parameters(), lr=float(args.router_lr), weight_decay=float(args.router_wd)
        )

    # --- Teacher target ---
    # fixed_env uses the original per-env random linear map teacher.
    # game_router uses a per-game mixture teacher (with a fixed gating rule) to supervise routing.
    teacher_env: Optional[SyntheticEnvTeacher] = None
    teacher_A: Optional[torch.Tensor] = None
    teacher_game: Optional[FixedGameTeacher] = None

    if args.routing_mode == "fixed_env":
        teacher_env = SyntheticEnvTeacher(args.n_envs, args.d_model, seed=args.seed, noise_std=0.0)
        teacher_A = teacher_env.A
        if device.type == "cuda":
            teacher_A = teacher_A.to(device=device, dtype=torch.float32)
    else:
        n_games = len(src.envs)
        k = int(args.experts_per_game)
        if args.n_experts < n_games * k:
            raise ValueError(
                f"routing_mode=game_router requires n_experts >= n_games*experts_per_game "
                f"({args.n_experts} < {n_games}*{k}={n_games*k}). "
                f"Either increase --n_experts or reduce --games/--experts_per_game."
            )
        teacher_game = FixedGameTeacher(
            n_games=n_games, k=k, d_model=args.d_model, seed=args.seed + 777, device=device
        )

    if args.sampler == "episode":
        sampler = EnvEpisodeSampler(args.n_envs, episode_len=args.episode_len, seed=args.seed + 1)
    else:
        sampler = EnvMarkovSampler(args.n_envs, p_stay=args.p_stay, seed=args.seed + 1)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 2)

    # --- Run logging / checkpointing ---
    metrics_logger = None
    start_step = 1
    if run_dir is not None:
        atomic_write_json(run_dir / "config.json", vars(args))
        metrics_logger = JsonlLogger(run_dir / "metrics.jsonl")

        if is_colab() and str(run_dir).startswith("/content/drive"):
            if not Path("/content/drive/MyDrive").exists():
                print(
                    "[warn] run_dir is under /content/drive but Google Drive doesn't appear mounted. "
                    "In a notebook cell run: from google.colab import drive; drive.mount('/content/drive')"
                )

        if args.resume:
            ckpt_path = run_dir / "checkpoint_last.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--resume set but checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            start_step = int(ckpt.get("step", 0)) + 1
            if "torch_rng" in ckpt:
                torch.set_rng_state(ckpt["torch_rng"])
            if device.type == "cuda" and "cuda_rng" in ckpt:
                try:
                    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
                except Exception:
                    pass
            if "numpy_rng" in ckpt and ckpt["numpy_rng"] is not None:
                try:
                    np.random.set_state(ckpt["numpy_rng"])
                except Exception:
                    pass
            if "g_state" in ckpt and ckpt["g_state"] is not None:
                g.set_state(ckpt["g_state"])
            if "sampler_state" in ckpt and ckpt["sampler_state"] is not None:
                try:
                    sampler.load_state_dict(ckpt["sampler_state"])
                except Exception:
                    pass
            if encoder is not None and "encoder_state" in ckpt and ckpt["encoder_state"] is not None:
                try:
                    encoder.load_state_dict(ckpt["encoder_state"])
                except Exception:
                    pass

            if router is not None and "router_state" in ckpt and ckpt["router_state"] is not None:
                try:
                    router.load_state_dict(ckpt["router_state"])
                except Exception:
                    pass
            if router_optim is not None and "router_optim" in ckpt and ckpt["router_optim"] is not None:
                try:
                    router_optim.load_state_dict(ckpt["router_optim"])
                except Exception:
                    pass

            if enc_optim is not None and "encoder_optim" in ckpt and ckpt["encoder_optim"] is not None:
                try:
                    enc_optim.load_state_dict(ckpt["encoder_optim"])
                except Exception:
                    pass

    tqdm = None if args.no_tqdm else _try_tqdm()
    iterator = range(start_step, args.steps + 1) if tqdm is None else tqdm(range(start_step, args.steps + 1))

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
            chunk = src.sample_from_buffer(int(eid), int(idx.shape[0]))
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

    def games_for_env_ids(env_ids: torch.Tensor) -> list[int]:
        env_cpu = env_ids.detach().to("cpu")
        return sorted({int(src.env_id_to_game_index(int(e))) for e in torch.unique(env_cpu).tolist()})

    def candidate_experts_for_env_ids(env_ids: torch.Tensor) -> list[int]:
        if args.routing_mode == "fixed_env":
            return torch.unique(route_fixed_by_env(env_ids, n_experts=args.n_experts)).tolist()
        # game_router: prefetch the whole expert block for each game in the batch
        n_games = len(src.envs)
        k = int(args.experts_per_game)
        game_ids = torch.unique((env_ids % n_games).long()).tolist()
        out: list[int] = []
        for gi in game_ids:
            base = int(gi) * k
            out.extend(list(range(base, base + k)))
        return out


    # Lookahead buffers for prefetch
    cur_env_ids = make_env_ids()
    cur_expert_ids = route_fixed_by_env(cur_env_ids, n_experts=args.n_experts)

    next_env_ids = make_env_ids()
    next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
    if args.prefetch:
        store.prefetch(candidate_experts_for_env_ids(next_env_ids))
    if args.prefetch_gpu:
        store.prefetch_to_gpu(candidate_experts_for_env_ids(next_env_ids), max_items=args.prefetch_gpu_max)

    t0_all = time.perf_counter()
    frames = 0


    for step in iterator:
        # Drain any completed prefetch work.
        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)

        # Launch GPU prefetch for *next-step candidates* into remaining free slots.
        if args.prefetch_gpu:
            store.prefetch_to_gpu(candidate_experts_for_env_ids(next_env_ids), max_items=args.prefetch_gpu_max)

        # --- Collect observations ---
        games_used = games_for_env_ids(cur_env_ids)
        obs = sample_obs_for_env_ids(cur_env_ids)
        frames += int(obs.shape[0])

        # Kick off environment stepping in the background to overlap with compute.
        if not args.no_pipeline_env:
            src.step_send(games_used)

        # Teacher features are fixed.
        x_teacher = teacher_feat(obs)

        # Compute y_true and (optionally) routing labels.
        router_labels: Optional[torch.Tensor] = None
        if args.routing_mode == "fixed_env":
            assert teacher_A is not None
            if teacher_A.device == cur_env_ids.device:
                A_batch = teacher_A.index_select(0, cur_env_ids)
            else:
                A_batch = teacher_A.index_select(0, cur_env_ids.to("cpu")).to(device=device, dtype=torch.float32)
            y_true = torch.einsum("bd,bde->be", x_teacher, A_batch)
            if args.teacher_noise > 0:
                y_true = y_true + torch.randn_like(y_true) * float(args.teacher_noise)
        else:
            assert teacher_game is not None
            n_games = len(src.envs)
            game_ids = (cur_env_ids % n_games).long()
            router_labels, y_true = teacher_game.labels_and_targets(
                x_teacher, game_ids, noise_std=float(args.teacher_noise)
            )

        # Student input to experts/router
        if encoder is None:
            x_in = x_teacher
        else:
            x_in = encoder(obs)

        # --- Routing (choose expert id per token) ---
        router_loss: Optional[torch.Tensor] = None
        if args.routing_mode == "fixed_env":
            cur_expert_ids = route_fixed_by_env(cur_env_ids, n_experts=args.n_experts)
        else:
            assert router is not None
            assert router_optim is not None
            assert router_labels is not None
            n_games = len(src.envs)
            k = int(args.experts_per_game)
            game_ids = (cur_env_ids % n_games).long()

            expert_ids = torch.empty((x_in.shape[0],), device=device, dtype=torch.long)
            router_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
            n_groups = 0

            miss_pen = float(args.route_miss_penalty)

            for gi in torch.unique(game_ids).tolist():
                idx = (game_ids == int(gi)).nonzero(as_tuple=False).squeeze(-1)
                xg = x_in.index_select(0, idx)

                logits = router(xg, int(gi))  # [n_i, k]

                # Cache-aware routing bias: subtract miss_cost(expert) * alpha from logits.
                if miss_pen > 0:
                    base = int(gi) * k
                    costs = []
                    for j in range(k):
                        eid = base + j
                        tier = store.residency(eid)
                        if tier == "gpu":
                            costs.append(float(args.miss_cost_gpu))
                        elif tier == "cpu":
                            costs.append(float(args.miss_cost_cpu))
                        else:
                            costs.append(float(args.miss_cost_disk))
                    cost_t = torch.tensor(costs, device=logits.device, dtype=logits.dtype)
                    logits = logits - miss_pen * cost_t.view(1, -1)

                temp = float(args.route_temperature)
                if temp != 1.0:
                    logits = logits / max(1e-6, temp)

                # Selection (argmax by default; sampling if requested).
                if args.route_sample:
                    probs = torch.softmax(logits, dim=-1)
                    local = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    local = torch.argmax(logits, dim=-1)

                expert_ids.index_copy_(0, idx, local + int(gi) * k)

                # Router supervision: match the teacher's within-game regime label.
                lab_g = router_labels.index_select(0, idx)
                router_loss_accum = router_loss_accum + F.cross_entropy(logits.float(), lab_g)
                n_groups += 1

            router_loss = router_loss_accum / max(1, int(n_groups))
            cur_expert_ids = expert_ids

        # DType shims
        if device.type == "cuda":
            x_in = x_in.to(dtype=compute_dtype)
            y_true = y_true.to(dtype=compute_dtype)

        # Enforce GPU slot budget and ensure experts are resident on GPU.
        n_unique = int(torch.unique(cur_expert_ids).numel())
        if n_unique > args.gpu_slots:
            raise RuntimeError(
                f"Batch uses {n_unique} unique experts but gpu_slots={args.gpu_slots}. "
                f"Increase --gpu_slots or reduce --envs_per_batch / batch_size / experts_per_game."
            )
        for eid in torch.unique(cur_expert_ids).tolist():
            store.ensure_on_gpu(int(eid))

        # Zero grads
        if enc_optim is not None:
            enc_optim.zero_grad(set_to_none=True)
        if router_optim is not None:
            router_optim.zero_grad(set_to_none=True)

        y_pred = moe_forward_top1(x_in, cur_expert_ids, store, sort_by_expert=args.sort_by_expert)
        mse = torch.mean((y_pred - y_true) ** 2)
        if router_loss is None:
            loss = mse
        else:
            loss = mse + float(args.router_loss_weight) * router_loss

        loss.backward()

        # Update experts out-of-core.
        for eid in torch.unique(cur_expert_ids).tolist():
            store.step_expert(int(eid), lr=args.lr, weight_decay=args.weight_decay)

        # Update encoder in-core (optional).
        if enc_optim is not None:
            enc_optim.step()

        # Update router in-core (optional).
        if router_optim is not None:
            router_optim.step()

        # Complete environment stepping (recv) after compute to overlap with the above.
        if not args.no_pipeline_env:
            src.step_recv(games_used)

        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)

        if device.type == "cuda" and (step % args.log_every == 0 or step == 1):
            torch.cuda.synchronize()

        if step % args.log_every == 0 or step == 1:
            loss_val = float(loss.detach().cpu().item())
            mse_val = float(mse.detach().cpu().item())
            router_loss_val = float(router_loss.detach().cpu().item()) if router_loss is not None else None

            cur_e0 = int(cur_env_ids[0].detach().cpu().item())
            game_idx = src.env_id_to_game_index(cur_e0)
            game_name = src.env_ids[game_idx]
            msg = (
                f"step={step:5d} loss={loss_val:.6f} mse={mse_val:.6f} "
                f"unique_experts={n_unique} gpu_cache={len(store.gpu_map)} cpu_cache={len(store.cpu_cache)} "
                f"env0={cur_e0} game={game_name}"
            )
            if router_loss_val is not None:
                msg += f" router_loss={router_loss_val:.4f}"
            if tqdm is None:
                print(msg)
            else:
                iterator.set_description(msg)

            if metrics_logger is not None:
                metrics_logger.log(
                    {
                        "step": int(step),
                        "loss": float(loss_val),
                        "mse": float(mse_val),
                        "router_loss": (float(router_loss_val) if router_loss_val is not None else None),
                        "unique_experts": int(n_unique),
                        "gpu_cache_size": int(len(store.gpu_map)),
                        "cpu_cache_size": int(len(store.cpu_cache)),
                        "frames": int(frames),
                        "env0": int(cur_e0),
                        "game": str(game_name),
                    }
                )

        # Shift lookahead and schedule prefetch.
        cur_env_ids, cur_expert_ids = next_env_ids, next_expert_ids
        next_env_ids = make_env_ids()
        next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
        if args.prefetch:
            store.prefetch(candidate_experts_for_env_ids(next_env_ids))

        # Periodic checkpoint (flush dirty experts to disk then save a lightweight run checkpoint).
        if run_dir is not None and args.checkpoint_every > 0 and (step % args.checkpoint_every == 0):
            store.flush_dirty(sync=True)
            ckpt = {
                "step": int(step),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "numpy_rng": np.random.get_state(),
                "g_state": g.get_state(),
                "sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
                "encoder_state": encoder.state_dict() if encoder is not None else None,
                "encoder_optim": enc_optim.state_dict() if enc_optim is not None else None,
                "router_state": router.state_dict() if router is not None else None,
                "router_optim": router_optim.state_dict() if router_optim is not None else None,
                "stats": stats.to_dict(),
            }
            atomic_torch_save(run_dir / "checkpoint_last.pt", ckpt)
            atomic_write_json(run_dir / "stats_latest.json", stats.to_dict())

    t_all = time.perf_counter() - t0_all
    if tqdm is not None:
        iterator.close()
    print(f"done in {t_all:.2f}s, approx frames={frames} ({frames / max(1e-9, t_all):.1f} frames/s)")

    src.close()
    store.flush_all()
    print()
    print(stats.summary())

    if run_dir is not None:
        atomic_write_json(run_dir / "stats_final.json", stats.to_dict())
        (run_dir / "stats_final.txt").write_text(stats.summary() + "\n")
        # Final checkpoint (expert state is stored in disk_root; this captures seeds + encoder parameters).
        ckpt = {
            "step": int(args.steps),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_rng": np.random.get_state(),
            "g_state": g.get_state(),
            "sampler_state": sampler.state_dict() if hasattr(sampler, "state_dict") else None,
            "encoder_state": encoder.state_dict() if encoder is not None else None,
            "encoder_optim": enc_optim.state_dict() if enc_optim is not None else None,
            "router_state": router.state_dict() if router is not None else None,
            "router_optim": router_optim.state_dict() if router_optim is not None else None,
            "stats": stats.to_dict(),
        }
        atomic_torch_save(run_dir / "checkpoint_last.pt", ckpt)

    if metrics_logger is not None:
        metrics_logger.close()


if __name__ == "__main__":
    main()

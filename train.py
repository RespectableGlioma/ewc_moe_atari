#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

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


def make_env_ids(
    *,
    sampler,
    envs_per_batch: int,
    batch_size: int,
    device: torch.device,
    g: torch.Generator,
) -> torch.Tensor:
    """Create per-token env ids with controllable "within-batch" diversity."""

    envs = [sampler.next() for _ in range(envs_per_batch)]
    env_ids = torch.tensor(
        [envs[int(torch.randint(0, len(envs), (1,), generator=g).item())] for _ in range(batch_size)],
        device=device,
        dtype=torch.long,
    )
    return env_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Out-of-core MoE training prototype (HBM<->DRAM<->Disk).")
    ap.add_argument("--n_experts", type=int, default=256)
    ap.add_argument("--n_envs", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--gpu_slots", type=int, default=16, help="HBM cache capacity (number of experts).")
    ap.add_argument(
        "--cpu_cache",
        type=int,
        default=64,
        help="DRAM cache capacity (number of experts). 0 => write-through to disk (requires --disk_root).",
    )
    ap.add_argument("--disk_root", type=str, default="", help="If set, use this directory as cold storage (NVMe).")
    ap.add_argument("--reset_disk", action="store_true", help="Delete --disk_root contents before running.")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", help="Compute dtype for experts on GPU: bf16/fp16/fp32")
    ap.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])

    ap.add_argument("--sampler", type=str, default="episode", choices=["episode", "markov"])
    ap.add_argument("--episode_len", type=int, default=20)
    ap.add_argument("--p_stay", type=float, default=0.9)
    ap.add_argument("--envs_per_batch", type=int, default=1, help="How many distinct envs per batch (affects unique experts).")

    ap.add_argument("--sort_by_expert", action="store_true", help="Sort tokens by expert in MoE forward (better locality).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)

    # Optimizer / out-of-core knobs
    ap.add_argument("--optim", type=str, default="adamw", choices=["sgd", "sgd_momentum", "adamw"])
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (only used for --optim sgd_momentum).")
    ap.add_argument("--adam_beta1", type=float, default=0.9)
    ap.add_argument("--adam_beta2", type=float, default=0.999)
    ap.add_argument("--adam_eps", type=float, default=1e-8)

    ap.add_argument(
        "--writeback_policy",
        type=str,
        default="evict",
        choices=["evict", "periodic", "writethrough"],
        help="When to write dirty CPU expert state to disk.",
    )
    ap.add_argument(
        "--writeback_every",
        type=int,
        default=0,
        help="For --writeback_policy periodic: flush dirty experts every N optimizer steps (sync).",
    )
    ap.add_argument("--io_workers", type=int, default=2, help="Background disk I/O threads (prefetch + eviction writeback).")
    ap.add_argument("--prefetch", action="store_true", help="Enable 1-step lookahead prefetch (disk->CPU).")
    ap.add_argument(
        "--prefetch_gpu",
        action="store_true",
        help="Enable 1-step lookahead GPU prefetch into free slots (CPU->GPU on a separate CUDA stream).",
    )
    ap.add_argument(
        "--prefetch_gpu_max",
        type=int,
        default=0,
        help="Max experts to GPU-prefetch per step. 0 => fill all free slots.",
    )

    # Optional latency simulation knobs (useful on fast local SSDs)
    ap.add_argument("--sim_h2d_gbps", type=float, default=0.0)
    ap.add_argument("--sim_d2h_gbps", type=float, default=0.0)
    ap.add_argument("--sim_disk_read_gbps", type=float, default=0.0)
    ap.add_argument("--sim_disk_write_gbps", type=float, default=0.0)
    ap.add_argument("--sim_extra_ms_per_io", type=float, default=0.0)

    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar.")

    args = ap.parse_args()

    torch.manual_seed(args.seed)

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

    teacher = SyntheticEnvTeacher(args.n_envs, args.d_model, seed=args.seed, noise_std=0.01)
    if args.sampler == "episode":
        sampler = EnvEpisodeSampler(args.n_envs, episode_len=args.episode_len, seed=args.seed + 1)
    else:
        sampler = EnvMarkovSampler(args.n_envs, p_stay=args.p_stay, seed=args.seed + 1)

    # Token-level selection among a small env set per batch.
    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 2)

    # tqdm setup
    tqdm = None if args.no_tqdm else _try_tqdm()
    iterator = range(1, args.steps + 1) if tqdm is None else tqdm(range(1, args.steps + 1))

    # Lookahead buffers for prefetch
    cur_env_ids = make_env_ids(
        sampler=sampler,
        envs_per_batch=args.envs_per_batch,
        batch_size=args.batch_size,
        device=device,
        g=g,
    )
    cur_expert_ids = route_fixed_by_env(cur_env_ids, n_experts=args.n_experts)

    next_env_ids = make_env_ids(
        sampler=sampler,
        envs_per_batch=args.envs_per_batch,
        batch_size=args.batch_size,
        device=device,
        g=g,
    )
    next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
    if args.prefetch:
        store.prefetch(torch.unique(next_expert_ids).tolist())
    if args.prefetch_gpu:
        # Best-effort; will only prefetch if CPU state is already available.
        store.prefetch_to_gpu(torch.unique(next_expert_ids).tolist(), max_items=args.prefetch_gpu_max)

    t0_all = time.perf_counter()

    for step in iterator:
        # Safety: don't exceed gpu_slots unique experts per step (autograd would break if we evict mid-step).
        n_unique = int(torch.unique(cur_expert_ids).numel())
        if n_unique > args.gpu_slots:
            raise RuntimeError(
                f"Batch uses {n_unique} unique experts but gpu_slots={args.gpu_slots}. "
                f"Increase --gpu_slots or reduce --envs_per_batch / batch_size."
            )

        # 0) Drain any completed prefetch work (disk->CPU and CPU->GPU) and
        #    ensure current experts are resident before we start autograd.
        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)
        for eid in torch.unique(cur_expert_ids).tolist():
            store.ensure_on_gpu(int(eid))

        # 0.5) Launch GPU prefetch for next-step experts into any remaining free slots.
        if args.prefetch_gpu:
            store.prefetch_to_gpu(torch.unique(next_expert_ids).tolist(), max_items=args.prefetch_gpu_max)

        x, y_true = teacher.sample_per_token(cur_env_ids, device=device, dtype=torch.float32)
        if device.type == "cuda":
            x = x.to(dtype=compute_dtype)
            y_true = y_true.to(dtype=compute_dtype)

        # Forward/Backward
        y_pred = moe_forward_top1(x, cur_expert_ids, store, sort_by_expert=args.sort_by_expert)
        loss = torch.mean((y_pred - y_true) ** 2)
        loss.backward()

        # Optimizer update per active expert (out-of-core: grads GPU->CPU, CPU update, CPU->GPU sync)
        for eid in torch.unique(cur_expert_ids).tolist():
            store.step_expert(int(eid), lr=args.lr, weight_decay=args.weight_decay)

        # Drain any completed prefetch work.
        store.drain_prefetch(max_items=64)
        store.drain_gpu_prefetch(max_items=64)

        if device.type == "cuda" and (step % args.log_every == 0 or step == 1):
            torch.cuda.synchronize()

        if step % args.log_every == 0 or step == 1:
            loss_val = float(loss.detach().cpu().item())
            msg = (
                f"step={step:5d} loss={loss_val:.6f} unique_experts={n_unique} "
                f"gpu_cache={len(store.gpu_map)} cpu_cache={len(store.cpu_cache)}"
            )
            if tqdm is None:
                print(msg)
            else:
                iterator.set_description(msg)

        # Shift lookahead + schedule prefetch for the newly-sampled next batch
        cur_env_ids, cur_expert_ids = next_env_ids, next_expert_ids
        next_env_ids = make_env_ids(
            sampler=sampler,
            envs_per_batch=args.envs_per_batch,
            batch_size=args.batch_size,
            device=device,
            g=g,
        )
        next_expert_ids = route_fixed_by_env(next_env_ids, n_experts=args.n_experts)
        if args.prefetch:
            store.prefetch(torch.unique(next_expert_ids).tolist())

    t_all = time.perf_counter() - t0_all
    if tqdm is None:
        print(f"done in {t_all:.2f}s")
    else:
        iterator.close()

    store.flush_all()
    print()
    print(stats.summary())


if __name__ == "__main__":
    main()

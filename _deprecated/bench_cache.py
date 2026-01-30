
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from ooc_moe.synthetic import EnvEpisodeSampler, EnvMarkovSampler
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


def main():
    ap = argparse.ArgumentParser(description="Benchmark the tiered expert cache without training compute.")
    ap.add_argument("--n_experts", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)

    ap.add_argument("--gpu_slots", type=int, default=16)
    ap.add_argument("--cpu_cache", type=int, default=64)
    ap.add_argument("--disk_root", type=str, default="", help="If set, use disk as cold store.")
    ap.add_argument("--reset_disk", action="store_true", help="Delete --disk_root contents before running.")

    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16")

    ap.add_argument("--sampler", type=str, default="episode", choices=["episode", "markov"])
    ap.add_argument("--episode_len", type=int, default=20)
    ap.add_argument("--p_stay", type=float, default=0.9)

    ap.add_argument("--experts_per_step", type=int, default=4, help="How many experts are requested per step.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)

    # latency sim
    ap.add_argument("--sim_h2d_gbps", type=float, default=12.0)
    ap.add_argument("--sim_d2h_gbps", type=float, default=12.0)
    ap.add_argument("--sim_disk_read_gbps", type=float, default=3.0)
    ap.add_argument("--sim_disk_write_gbps", type=float, default=2.0)
    ap.add_argument("--sim_extra_ms_per_io", type=float, default=0.2)

    args = ap.parse_args()

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
        activation="gelu",
        disk_root=disk_root,
        pin_cpu=True,
        optim="sgd",
        latency_sim=lat,
        stats=stats,
        seed=args.seed,
    )

    # Env stream used as a proxy for correlated access
    n_envs = args.n_experts
    if args.sampler == "episode":
        sampler = EnvEpisodeSampler(n_envs, episode_len=args.episode_len, seed=args.seed + 1)
    else:
        sampler = EnvMarkovSampler(n_envs, p_stay=args.p_stay, seed=args.seed + 1)

    g = torch.Generator(device="cpu")
    g.manual_seed(args.seed + 2)

    for step in range(1, args.steps + 1):
        # request a small set of experts with some locality around current env id
        base = sampler.next()
        # choose experts_per_step experts near 'base'
        eids = []
        for i in range(args.experts_per_step):
            jitter = int(torch.randint(0, 8, (1,), generator=g).item())
            eids.append((base + jitter) % args.n_experts)

        # touch them
        for eid in sorted(set(eids)):
            store.ensure_on_gpu(int(eid))

        if step % args.log_every == 0 or step == 1:
            print(f"step={step:5d} gpu_cache={len(store.gpu_map)} cpu_cache={len(store.cpu_cache)}")

    store.flush_all()
    print()
    print(stats.summary())


if __name__ == "__main__":
    main()

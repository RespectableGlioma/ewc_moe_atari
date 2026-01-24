
# Out-of-core MoE Expert Cache (HBM ↔ DRAM ↔ Disk) — Prototype

This is a small research **prototype codebase** to experiment with *out-of-core training ideas* using a sparse top‑1 MoE layer and a **tiered expert cache**:

- **Tier 0 (HBM):** a fixed number of GPU “slots” (experts resident on GPU)
- **Tier 1 (DRAM):** an LRU warm cache of expert parameters on CPU
- **Tier 2 (Disk/NVMe):** a per‑expert file store (cold storage)

The goal is to make it easy to test hypotheses like:
- “If data is environment-correlated, can we keep a small expert working set hot?”
- “What cache sizes do we need to avoid thrashing?”
- “What is the cost of H2D/D2H and disk I/O under different locality patterns?”

> **Important limitation:**  
> This prototype assumes the number of **unique experts used in one step** is ≤ `gpu_slots`.  
> Otherwise you’d need to evict/overwrite weights before backward, which will break autograd.

---

## Files

- `train.py` — simple synthetic training loop (MSE regression teacher per environment)
- `train_ale.py` — training loop on real Atari observations (ALE / Gymnasium)
- `bench_cache.py` — stress-test the cache without training compute
- `ooc_moe/tiered_store.py` — the tiered cache + disk store + writeback
- `ooc_moe/moe.py` — top‑1 MoE forward + fixed router
- `ooc_moe/synthetic.py` — environment-correlated stream + synthetic teacher
- `ooc_moe/expert.py` — functional FFN expert + GPU slot tensors
- `ooc_moe/atari.py` — Atari (ALE) vector env batching helper

---

## Quickstart

### 1) Run cache-only benchmark (recommended first)
This uses an **artificial latency simulator** (so you can “feel” the effect of misses even on a fast machine).

```bash
python bench_cache.py \
  --n_experts 2048 \
  --gpu_slots 16 \
  --cpu_cache 64 \
  --disk_root /tmp/ooc_disk \
  --sampler episode --episode_len 50 \
  --experts_per_step 4 \
  --sim_h2d_gbps 12 --sim_d2h_gbps 12 \
  --sim_disk_read_gbps 3 --sim_disk_write_gbps 2 \
  --sim_extra_ms_per_io 0.2
```

Look at:
- `gpu_cache_hit / gpu_cache_miss`
- `cpu_cache_hit / cpu_cache_miss`
- `disk_read_ops`, `disk_write_ops`
- `h2d_load`, `h2d_sync`, `d2h_grads`, `disk_read`, `disk_write` timers

---

### 2) Run training demo (synthetic teacher, out-of-core optimizer state)
```bash
python train.py \
  --n_experts 512 --n_envs 512 \
  --gpu_slots 16 --cpu_cache 64 \
  --disk_root /tmp/ooc_disk \
  --sampler episode --episode_len 30 \
  --envs_per_batch 1 \
  --batch_size 1024 --steps 200 \
  --lr 1e-2 --optim adamw \
  --prefetch --io_workers 2 \
  --writeback_policy evict \
  --sort_by_expert
```

Notes:
- `--envs_per_batch 1` keeps one environment per batch → strong locality.
- Increasing `--episode_len` increases locality across steps.
- Increasing `--envs_per_batch` increases unique experts per step.

---

## How the tiered store works (v1 semantics)

This version is oriented around *real out-of-core training mechanics*:

- **CPU is canonical** for expert parameters (FP32 master weights) and optimizer state (momentum/Adam).
- **GPU slots are a cache** used only for forward/backward compute.
- **Disk/NVMe is the cold backing store** when experts are not present in the warm CPU cache.

One training step (for the active experts) looks like:

1. `ensure_on_gpu(eid)` loads expert weights CPU→GPU (HBM cache).
2. forward/backward runs on GPU.
3. `step_expert(eid)`:
   - copies grads GPU→CPU (`d2h_grads`)
   - updates CPU master weights + optimizer state (SGD/SGD-momentum/AdamW)
   - syncs updated weights CPU→GPU (`h2d_sync`) so reusing the cached expert is correct
4. Dirty CPU experts are persisted according to `--writeback_policy`:
   - `evict` (default): write back only when the CPU LRU evicts an expert (async)
   - `periodic`: sync flush dirty experts every N steps
   - `writethrough`: sync flush every step (slow, but simplest correctness)

Prefetch:
- `--prefetch` triggers a 1-step lookahead disk→CPU prefetch of the next batch’s expert IDs.
- Completed reads are inserted into the CPU LRU by `drain_prefetch()`.

GPU prefetch:
- `--prefetch_gpu` triggers a 1-step lookahead CPU→GPU prefetch into *currently free* HBM slots using a separate CUDA stream.
  - It never evicts GPU-resident experts (safe w.r.t. autograd).
  - It only prefetches experts whose CPU state is already available (CPU-cache hit or completed disk-prefetch staged in a small stash) so it does not block the training thread.

> Note: `--cpu_cache 0` now forces safe write-through to disk during `step_expert()` so training remains correct (but will be much slower than having even a small DRAM cache).

---

## Next steps (what to build next)

This is the skeleton for the bigger ideas we discussed:
1. **GPU prefetch**: prefetch into *free* HBM slots for the next microbatch (never evict mid-step).
2. **Cache-aware routing**: add a routing penalty to prefer experts already resident (with load-balancing).
3. **Hot/cold optimizer policies**: keep full AdamW for hot experts; cheaper/lazy updates for cold experts.
4. **Evict-safe backward**: activation checkpoint + recompute to allow unique experts per step > `gpu_slots`.
5. **Real NVMe I/O**: replace `torch.save/torch.load` with an async tensor I/O backend (AIO/GDS).

---

## Atari/ALE integration (real observations)

This repo now includes a **real-observation** training script, `train_ale.py`, that
keeps the same out-of-core expert/optimizer machinery but replaces the synthetic
`x ~ N(0,I)` stream with **Atari frames**.

### Install Atari deps

Gymnasium v1.0 removed the hidden “plugin import” behavior that used to let you
create Atari envs without importing `ale_py` first. Now, you either:

- import `ale_py` and call `gym.register_envs(ale_py)` before `gym.make(...)`, or
- use the `module:env_id` form like `ale_py:ALE/Pong-v5`.

This codebase handles that internally, but you still need the packages.

Recommended:

```bash
pip install "gymnasium[atari]"
```

Or, using the provided file:

```bash
pip install -r requirements_atari.txt
```

### Run the Atari demo

Fastest backend (ALE's native C++ vector env via `gym.make_vec`, which the ALE docs describe as
equivalent to `AtariPreprocessing` + `FrameStackObservation`):

```bash
python train_ale.py \
  --n_experts 512 --n_envs 512 \
  --gpu_slots 16 --cpu_cache 64 \
  --disk_root /tmp/ooc_disk --reset_disk \
  --games Pong,Breakout,SpaceInvaders,Seaquest \
  --env_backend ale_vec --vec_envs_per_game 8 \
  --batch_size 512 --steps 200 \
  --lr 1e-2 --optim adamw \
  --prefetch --prefetch_gpu --io_workers 4 \
  --writeback_policy evict \
  --sort_by_expert
```

Pure-Gymnasium wrapper backend (explicit `AtariPreprocessing` + `FrameStackObservation`):

```bash
python train_ale.py \
  --env_backend wrappers \
  --games Pong,Breakout \
  --batch_size 256 --steps 100
```

Notes:

- By default, the “teacher” target is computed from a **fixed (non-trainable) featurizer** of the
  Atari frames. This prevents a trivial collapse where a learnable encoder outputs zeros and makes
  the objective vanish.
- If you want a learnable in-core encoder anyway, add `--encoder conv`.

If you tell me your target environment (A100/H100? PCIe vs NVLink? local NVMe?), we can evolve the prototype toward something closer to a real training system.

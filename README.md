
# Out-of-core MoE Expert Cache (HBM ↔ DRAM ↔ Disk) — Prototype

This is a small research **prototype codebase** to experiment with *out-of-core training ideas* using a sparse top‑1 MoE layer and a **tiered expert cache**:

- **Tier 0 (HBM):** a fixed number of GPU “slots” (experts resident on GPU)
- **Tier 1 (DRAM):** an LRU warm cache of expert parameters on CPU
- **Tier 2 (Disk/NVMe):** a per‑expert file store (cold storage)

The goal is to make it easy to test hypotheses like:
- “If data is environment-correlated, can we keep a small expert working set hot?”
- “What cache sizes do we need to avoid thrashing?”
- “What is the cost of H2D/D2H and disk I/O under different locality patterns?”

> **Important limitation (v0):**  
> This prototype assumes the number of **unique experts used in one step** is ≤ `gpu_slots`.  
> Otherwise you’d need to evict/overwrite weights before backward, which will break autograd.

---

## Files

- `train.py` — simple synthetic training loop (MSE regression teacher per environment)
- `bench_cache.py` — stress-test the cache without training compute
- `ooc_moe/tiered_store.py` — the tiered cache + disk store + writeback
- `ooc_moe/moe.py` — top‑1 MoE forward + fixed router
- `ooc_moe/synthetic.py` — environment-correlated stream + synthetic teacher
- `ooc_moe/expert.py` — functional FFN expert + GPU slot tensors

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
- `h2d_load`, `d2h_writeback`, `disk_read`, `disk_write` timers

---

### 2) Run training demo (synthetic teacher)
```bash
python train.py \
  --n_experts 512 --n_envs 512 \
  --gpu_slots 16 --cpu_cache 64 \
  --disk_root /tmp/ooc_disk \
  --sampler episode --episode_len 30 \
  --envs_per_batch 1 \
  --batch_size 1024 --steps 200 \
  --lr 1e-2 \
  --sort_by_expert
```

Notes:
- `--envs_per_batch 1` keeps one environment per batch → strong locality.
- Increasing `--episode_len` increases locality across steps.
- Increasing `--envs_per_batch` increases unique experts per step.

---

## How the tiered store works (v0 semantics)

- When an expert is loaded into a **GPU slot**, the GPU copy is treated as canonical.
- After `loss.backward()`, we do **SGD update in-place on the GPU** (`TieredExpertStore.sgd_step_inplace`).
- When we need a slot for a different expert, we **evict** the LRU resident expert and **write back** its updated weights:
  - GPU → CPU warm cache (D2H)
  - and if warm cache evicts, CPU → disk (cold)

This approximates: **HBM hot set**, **DRAM warm set**, **disk cold set**.

---

## Next steps (what to build next)

This is the skeleton for the bigger ideas we discussed:
1. **Prefetch**: predict the next expert set and asynchronously load from disk→CPU→GPU.
2. **Cache-aware routing**: add a penalty to routing to prefer experts already resident (subject to load balancing).
3. **Optimizer state**: add momentum/Adam states and decide where they live (HBM vs DRAM vs disk).
4. **Evict-safe backward**: checkpoint + recompute on backward to allow unique experts per step > `gpu_slots`.
5. **Real NVMe + AIO/GDS**: replace `torch.save/torch.load` with a tensor I/O backend.

If you tell me your target environment (A100/H100? PCIe vs NVLink? local NVMe?), we can evolve the prototype toward something closer to a real training system.

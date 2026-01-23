
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .expert import ExpertState, ExpertSlot, make_expert_state
from .utils import LRUCache, Stats, timed


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def state_nbytes(state: ExpertState) -> int:
    nb = 0
    for t in state.params.values():
        nb += tensor_nbytes(t)
    if state.opt:
        for t in state.opt.values():
            nb += tensor_nbytes(t)
    return nb


@dataclass
class LatencySim:
    """
    Optional latency/bandwidth simulator to make behavior visible even on fast disks.

    Set any field to 0.0 to disable.
    """
    h2d_gbps: float = 0.0  # host->device bandwidth in GB/s
    d2h_gbps: float = 0.0  # device->host bandwidth in GB/s
    disk_read_gbps: float = 0.0
    disk_write_gbps: float = 0.0
    extra_ms_per_io: float = 0.0  # fixed overhead per disk read/write

    def _sleep_for_bytes(self, nbytes: int, gbps: float, extra_ms: float = 0.0) -> None:
        if gbps <= 0.0:
            return
        seconds = (nbytes / 1e9) / gbps
        if extra_ms > 0:
            seconds += extra_ms / 1000.0
        if seconds > 0:
            time.sleep(seconds)

    def simulate_h2d(self, nbytes: int) -> None:
        self._sleep_for_bytes(nbytes, self.h2d_gbps)

    def simulate_d2h(self, nbytes: int) -> None:
        self._sleep_for_bytes(nbytes, self.d2h_gbps)

    def simulate_disk_read(self, nbytes: int) -> None:
        self._sleep_for_bytes(nbytes, self.disk_read_gbps, self.extra_ms_per_io)

    def simulate_disk_write(self, nbytes: int) -> None:
        self._sleep_for_bytes(nbytes, self.disk_write_gbps, self.extra_ms_per_io)


class DiskKVStore:
    """
    Simple per-expert file store.

    This is *not* a high-performance tensor store. It's a practical prototype.
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, expert_id: int) -> Path:
        return self.root / f"expert_{expert_id:08d}.pt"

    def exists(self, expert_id: int) -> bool:
        return self._path(expert_id).exists()

    def save(self, expert_id: int, state: ExpertState) -> int:
        path = self._path(expert_id)
        payload = {
            "params": {k: v.detach().to("cpu") for k, v in state.params.items()},
            "opt": {k: v.detach().to("cpu") for k, v in (state.opt or {}).items()},
        }
        torch.save(payload, path)
        return int(os.path.getsize(path))

    def load(self, expert_id: int) -> ExpertState:
        path = self._path(expert_id)
        payload = torch.load(path, map_location="cpu")
        params = payload["params"]
        opt = payload.get("opt", {}) or None
        if opt is not None and len(opt) == 0:
            opt = None
        return ExpertState(params=params, opt=opt)


@dataclass
class GPUResident:
    slot_idx: int
    dirty: bool = False


class TieredExpertStore:
    """
    HBM (GPU slots) <-> DRAM (CPU cache) <-> Disk (NVMe) expert store.

    Canonicality model (v0):
      - If expert is GPU-resident, GPU slot is canonical; it becomes dirty after SGD update.
      - On GPU eviction, we write back to CPU (warm cache) and possibly to disk (cold).
      - CPU cache is canonical for experts that are not in GPU.
      - Disk is canonical when an expert is neither in GPU nor CPU cache.

    This is intentionally simple; it's meant for experimentation.
    """
    def __init__(
        self,
        *,
        n_experts: int,
        d_model: int,
        d_hidden: int,
        gpu_slots: int,
        cpu_cache_capacity: int,
        device: torch.device,
        compute_dtype: torch.dtype,
        activation: str,
        disk_root: Optional[Path],
        pin_cpu: bool = True,
        with_momentum: bool = False,
        latency_sim: Optional[LatencySim] = None,
        stats: Optional[Stats] = None,
        seed: int = 0,
    ):
        if gpu_slots <= 0:
            raise ValueError("gpu_slots must be > 0")
        if cpu_cache_capacity < 0:
            raise ValueError("cpu_cache_capacity must be >= 0")

        self.n_experts = n_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.device = device
        self.compute_dtype = compute_dtype
        self.activation = activation
        self.pin_cpu = pin_cpu
        self.with_momentum = with_momentum

        self.stats = stats or Stats()
        self.lat = latency_sim or LatencySim()

        self.disk: Optional[DiskKVStore] = DiskKVStore(Path(disk_root)) if disk_root else None

        self.cpu_cache: LRUCache[int, ExpertState] = LRUCache(cpu_cache_capacity)
        self.gpu_map: LRUCache[int, GPUResident] = LRUCache(gpu_slots)

        self.slots = [
            ExpertSlot(d_model, d_hidden, device=device, compute_dtype=compute_dtype, activation=activation)
            for _ in range(gpu_slots)
        ]
        self.free_slots = set(range(gpu_slots))

        # Seed base used for lazy expert initialization
        self.seed_base = int(seed)
        self._bootstrap()

    def _bootstrap(self) -> None:
        """
        Lazy bootstrap.

        We *do not* materialize all experts up front, because that would dominate startup time
        at large n_experts. Experts are created on-demand the first time they are requested.
        """
        # Ensure disk directory exists if configured
        if self.disk is not None:
            self.disk.root.mkdir(parents=True, exist_ok=True)

    def _load_to_cpu(self, expert_id: int) -> ExpertState:
        # First try warm DRAM cache
        st = self.cpu_cache.get(expert_id)
        if st is not None:
            self.stats.inc("cpu_cache_hit", 1)
            return st
        self.stats.inc("cpu_cache_miss", 1)

        if self.disk is None:
            # No cold store. Create expert lazily and keep it in DRAM cache (if enabled).
            self.stats.inc("cpu_cache_miss_reinit", 1)
            st = make_expert_state(
                self.d_model, self.d_hidden,
                seed=self.seed_base + expert_id,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.pin_cpu,
                with_momentum=self.with_momentum,
            )
            if self.cpu_cache.capacity > 0:
                evicted, _ = self.cpu_cache.put(expert_id, st)
                # If DRAM cache is too small and there's no disk, evictions mean we lose state.
                if evicted is not None:
                    self.stats.inc("cpu_evictions_no_disk", 1)
            return st


        # If cold store is enabled but this expert hasn't been materialized yet, create it lazily.
        if self.disk is not None and not self.disk.exists(expert_id):
            st = make_expert_state(
                self.d_model, self.d_hidden,
                seed=self.seed_base + expert_id,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.pin_cpu,
                with_momentum=self.with_momentum,
            )
            # Persist initial state so cold loads are well-defined.
            with timed(self.stats, "disk_write"):
                nfile = self.disk.save(expert_id, st)
            nbytes = state_nbytes(st)
            self.stats.add_bytes("disk_write_bytes", nbytes)
            self.stats.inc("disk_write_ops", 1)
            self.lat.simulate_disk_write(nbytes)
            # Optionally warm-cache it
            if self.cpu_cache.capacity > 0:
                evicted, _ = self.cpu_cache.put(expert_id, st)
                if evicted is not None:
                    old_id, old_state = evicted
                    self._flush_cpu_to_disk(old_id, old_state)
                return st
            return st


        # Load from disk (cold)
        with timed(self.stats, "disk_read"):
            st = self.disk.load(expert_id)
        nbytes = state_nbytes(st)
        self.stats.add_bytes("disk_read_bytes", nbytes)
        self.stats.inc("disk_read_ops", 1)
        self.lat.simulate_disk_read(nbytes)

        # Optionally pin tensors to improve H2D
        if self.pin_cpu and torch.cuda.is_available():
            for k in list(st.params.keys()):
                if st.params[k].device.type == "cpu" and not st.params[k].is_pinned():
                    st.params[k] = st.params[k].pin_memory()
            if st.opt:
                for k in list(st.opt.keys()):
                    if st.opt[k].device.type == "cpu" and not st.opt[k].is_pinned():
                        st.opt[k] = st.opt[k].pin_memory()

        # Insert into CPU cache (warm) if enabled
        if self.cpu_cache.capacity > 0:
            evicted, _ = self.cpu_cache.put(expert_id, st)
            if evicted is not None:
                old_id, old_state = evicted
                self._flush_cpu_to_disk(old_id, old_state)
        return st

    def _flush_cpu_to_disk(self, expert_id: int, st: ExpertState) -> None:
        if self.disk is None:
            return
        with timed(self.stats, "disk_write"):
            nfile = self.disk.save(expert_id, st)
        nbytes = state_nbytes(st)
        self.stats.add_bytes("disk_write_bytes", nbytes)
        self.stats.inc("disk_write_ops", 1)
        self.lat.simulate_disk_write(nbytes)

    def _writeback_gpu_resident(self, expert_id: int, res: GPUResident) -> None:
        """
        Write back GPU slot parameters into CPU cache (and maybe disk).
        """
        slot = self.slots[res.slot_idx]
        # Copy weights to CPU float32 tensors
        with timed(self.stats, "d2h_writeback"):
            cpu_params: Dict[str, torch.Tensor] = {}
            nbytes = 0
            for k, t in slot.params.items():
                cpu_t = t.detach().to(device="cpu", dtype=torch.float32)
                if self.pin_cpu and torch.cuda.is_available() and not cpu_t.is_pinned():
                    cpu_t = cpu_t.pin_memory()
                cpu_params[k] = cpu_t
                nbytes += tensor_nbytes(cpu_t)
        self.stats.add_bytes("d2h_bytes", nbytes)
        self.stats.inc("d2h_ops", 1)
        self.lat.simulate_d2h(nbytes)

        st = ExpertState(params=cpu_params, opt=None)

        # Warm cache writeback (if enabled), otherwise flush directly to disk.
        if self.cpu_cache.capacity > 0:
            evicted, _ = self.cpu_cache.put(expert_id, st)
            if evicted is not None:
                old_id, old_state = evicted
                self._flush_cpu_to_disk(old_id, old_state)
        else:
            # No warm cache => write through to disk (if present).
            if self.disk is not None:
                self._flush_cpu_to_disk(expert_id, st)

    def ensure_on_gpu(self, expert_id: int) -> Tuple[ExpertSlot, int]:
        """
        Ensure expert is resident in a GPU slot. Returns (slot, slot_idx).
        """
        res = self.gpu_map.get(expert_id)
        if res is not None:
            self.stats.inc("gpu_cache_hit", 1)
            return self.slots[res.slot_idx], res.slot_idx

        self.stats.inc("gpu_cache_miss", 1)

        # Need CPU state to load (warm or cold)
        cpu_state = self._load_to_cpu(expert_id)

        # Choose a slot
        if self.free_slots:
            slot_idx = self.free_slots.pop()
        else:
            # Evict LRU from gpu_map (pop oldest)
            # OrderedDict stores LRU oldest at beginning. Our LRUCache doesn't expose popitem,
            # so emulate by taking first key.
            old_eid = next(iter(self.gpu_map.keys()))
            old_res = self.gpu_map.pop(old_eid)
            assert old_res is not None
            self.stats.inc("gpu_evictions", 1)

            # Writeback if dirty (in v0, always write back on eviction for simplicity).
            # (You can change this to conditional if you add clean/dirty tracking.)
            self._writeback_gpu_resident(old_eid, old_res)

            slot_idx = old_res.slot_idx

        slot = self.slots[slot_idx]

        with timed(self.stats, "h2d_load"):
            moved = slot.load_from_cpu(cpu_state, non_blocking=True)
        self.stats.add_bytes("h2d_bytes", moved)
        self.stats.inc("h2d_ops", 1)
        self.lat.simulate_h2d(moved)

        # Insert into GPU map
        self.gpu_map.put(expert_id, GPUResident(slot_idx=slot_idx, dirty=False))
        return slot, slot_idx

    @torch.no_grad()
    def sgd_step_inplace(self, expert_id: int, lr: float, weight_decay: float = 0.0) -> None:
        """
        Apply SGD update directly on the GPU-resident parameters (canonical while resident).
        """
        res = self.gpu_map.get(expert_id)
        if res is None:
            raise RuntimeError("Expert not on GPU; call ensure_on_gpu first.")
        slot = self.slots[res.slot_idx]

        for k, p in slot.params.items():
            if p.grad is None:
                continue
            g = p.grad
            if weight_decay != 0.0:
                g = g.add(p, alpha=weight_decay)
            p.add_(g, alpha=-lr)
        slot.zero_grad()
        # Mark dirty
        res.dirty = True
        # Re-put to update LRU position (already done via get, but keep canonical)
        self.gpu_map.put(expert_id, res)

    def flush_all(self) -> None:
        """
        Write back all GPU-resident experts to CPU and disk.
        """
        # Evict everything from GPU map
        for eid, res in list(self.gpu_map.items()):
            self._writeback_gpu_resident(eid, res)
            self.free_slots.add(res.slot_idx)
        self.gpu_map = LRUCache(self.gpu_map.capacity)

        # Flush CPU cache to disk if enabled
        if self.disk is not None:
            for eid, st in list(self.cpu_cache.items()):
                self._flush_cpu_to_disk(eid, st)

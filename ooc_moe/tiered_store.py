from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Dict, Iterable, Optional, Tuple

import torch

from .expert import ExpertOptState, ExpertState, ExpertSlot, TensorDict, make_expert_state
from .utils import LRUCache, Stats, timed


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def state_nbytes(state: ExpertState) -> int:
    nb = 0
    for t in state.params.values():
        nb += tensor_nbytes(t)
    if state.opt is not None:
        for t in state.opt.tensors.values():
            nb += tensor_nbytes(t)
    return nb


@dataclass
class LatencySim:
    """Optional latency/bandwidth simulator.

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
    """Simple per-expert file store.

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
            "opt": None,
        }
        if state.opt is not None:
            payload["opt"] = {
                "kind": state.opt.kind,
                "step": int(state.opt.step),
                "tensors": {k: v.detach().to("cpu") for k, v in state.opt.tensors.items()},
            }
        torch.save(payload, path)
        return int(os.path.getsize(path))

    def load(self, expert_id: int) -> ExpertState:
        path = self._path(expert_id)
        try:
            # Newer PyTorch versions warn on pickle by default; this keeps loads "tensor-only".
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch: weights_only may not exist.
            payload = torch.load(path, map_location="cpu")
        params = payload["params"]
        opt_payload = payload.get("opt", None)
        opt: Optional[ExpertOptState] = None
        if opt_payload is not None:
            opt = ExpertOptState(
                kind=str(opt_payload.get("kind", "sgd")),
                step=int(opt_payload.get("step", 0)),
                tensors=opt_payload.get("tensors", {}),
            )
        return ExpertState(params=params, opt=opt)


@dataclass
class CPUResident:
    state: ExpertState
    dirty: bool = False


@dataclass
class GPUResident:
    slot_idx: int


class TieredExpertStore:
    """HBM (GPU slots) <-> DRAM (CPU cache) <-> Disk (NVMe) expert store.

    v1 canonicality model:
      - CPU is canonical for params + optimizer state.
      - GPU slots are *read/compute caches* (weights copied in/out of CPU).
      - Disk is canonical when an expert is not in the CPU warm cache.

    This makes it straightforward to support out-of-core optimizer state:
      - Gradients are produced on GPU, copied to CPU.
      - The optimizer step runs on CPU using CPU-resident optimizer state.
      - Updated weights are copied back to the GPU slot *if the expert remains resident*.

    Prefetch:
      - We asynchronously load requested experts from disk into a "ready" queue.
      - The main thread periodically drains that queue and inserts into the CPU LRU.

    Writeback policy:
      - Dirty CPU-resident experts are written back on eviction (async).
      - Optionally, you can enable synchronous periodic flush or write-through.
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
        optim: str = "sgd",
        momentum: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        writeback_policy: str = "evict",  # "evict" | "periodic" | "writethrough"
        writeback_every: int = 0,
        io_workers: int = 2,
        latency_sim: Optional[LatencySim] = None,
        stats: Optional[Stats] = None,
        seed: int = 0,
    ):
        if gpu_slots <= 0:
            raise ValueError("gpu_slots must be > 0")
        if cpu_cache_capacity < 0:
            raise ValueError("cpu_cache_capacity must be >= 0")
        if optim not in ("sgd", "sgd_momentum", "adamw"):
            raise ValueError("optim must be one of: sgd, sgd_momentum, adamw")
        if writeback_policy not in ("evict", "periodic", "writethrough"):
            raise ValueError("writeback_policy must be one of: evict, periodic, writethrough")

        self.n_experts = int(n_experts)
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.device = device
        self.compute_dtype = compute_dtype
        self.activation = activation
        self.pin_cpu = bool(pin_cpu)

        self.optim = optim
        self.momentum = float(momentum)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)

        self.writeback_policy = writeback_policy
        self.writeback_every = int(writeback_every)
        self._step_counter = 0

        self.stats = stats or Stats()
        self.lat = latency_sim or LatencySim()

        self.disk: Optional[DiskKVStore] = DiskKVStore(Path(disk_root)) if disk_root else None
        if self.disk is not None:
            # Prevent confusing shape mismatches if the same disk_root is reused with different model dims.
            meta_path = self.disk.root / "meta.json"
            meta = {
                "version": 1,
                "d_model": int(d_model),
                "d_hidden": int(d_hidden),
                "activation": str(activation),
                "optim": str(optim),
            }
            if meta_path.exists():
                try:
                    old = json.loads(meta_path.read_text())
                except Exception:
                    old = {}
                if int(old.get("d_model", -1)) != meta["d_model"] or int(old.get("d_hidden", -1)) != meta["d_hidden"]:
                    raise ValueError(
                        f"disk_root metadata mismatch at {self.disk.root}. "
                        f"Found d_model={old.get('d_model')}, d_hidden={old.get('d_hidden')}; "
                        f"requested d_model={meta['d_model']}, d_hidden={meta['d_hidden']}. "
                        f"Use a new --disk_root or delete the existing directory."
                    )
            else:
                meta_path.write_text(json.dumps(meta, indent=2))
        if self.disk is None and cpu_cache_capacity == 0:
            raise ValueError("cpu_cache_capacity=0 requires disk_root for correctness (otherwise state is lost).")

        self.cpu_cache: LRUCache[int, CPUResident] = LRUCache(cpu_cache_capacity)
        self.gpu_map: LRUCache[int, GPUResident] = LRUCache(gpu_slots)

        self.slots = [
            ExpertSlot(d_model, d_hidden, device=device, compute_dtype=compute_dtype, activation=activation)
            for _ in range(gpu_slots)
        ]
        self.free_slots = set(range(gpu_slots))

        # Seed base used for lazy expert initialization
        self.seed_base = int(seed)

        # Async I/O machinery (prefetch reads, eviction writeback)
        self._io_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_futs: Dict[int, Future] = {}
        self._prefetch_ready: "SimpleQueue[Tuple[int, ExpertState]]" = SimpleQueue()
        self._writeback_futs: list[Future] = []

        if self.disk is not None and io_workers > 0:
            self._io_executor = ThreadPoolExecutor(max_workers=int(io_workers))

    # ----------------------------
    # CPU state load / store
    # ----------------------------

    def _maybe_pin(self, st: ExpertState) -> ExpertState:
        if not (self.pin_cpu and torch.cuda.is_available()):
            return st
        for k in list(st.params.keys()):
            t = st.params[k]
            if t.device.type == "cpu" and not t.is_pinned():
                st.params[k] = t.pin_memory()
        if st.opt is not None:
            for k in list(st.opt.tensors.keys()):
                t = st.opt.tensors[k]
                if t.device.type == "cpu" and not t.is_pinned():
                    st.opt.tensors[k] = t.pin_memory()
        return st

    def _disk_load_or_init_sync(self, expert_id: int) -> ExpertState:
        """Load expert from disk, or lazily initialize and persist."""
        assert self.disk is not None

        # Lazy init if missing
        if not self.disk.exists(expert_id):
            self.stats.inc("disk_init", 1)
            st = make_expert_state(
                self.d_model,
                self.d_hidden,
                seed=self.seed_base + expert_id,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.pin_cpu,
                with_momentum=False,
            )
            with timed(self.stats, "disk_write"):
                self.disk.save(expert_id, st)
            nbytes = state_nbytes(st)
            self.stats.add_bytes("disk_write_bytes", nbytes)
            self.stats.inc("disk_write_ops", 1)
            self.lat.simulate_disk_write(nbytes)
            return self._maybe_pin(st)

        with timed(self.stats, "disk_read"):
            st = self.disk.load(expert_id)
        nbytes = state_nbytes(st)
        self.stats.add_bytes("disk_read_bytes", nbytes)
        self.stats.inc("disk_read_ops", 1)
        self.lat.simulate_disk_read(nbytes)
        return self._maybe_pin(st)

    def _disk_load_or_init_worker(self, expert_id: int) -> ExpertState:
        """Worker-thread variant used for prefetch."""
        assert self.disk is not None

        if not self.disk.exists(expert_id):
            self.stats.inc("disk_init", 1)
            st = make_expert_state(
                self.d_model,
                self.d_hidden,
                seed=self.seed_base + expert_id,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.pin_cpu,
                with_momentum=False,
            )
            with timed(self.stats, "disk_write"):
                self.disk.save(expert_id, st)
            nbytes = state_nbytes(st)
            self.stats.add_bytes("disk_write_bytes", nbytes)
            self.stats.inc("disk_write_ops", 1)
            self.lat.simulate_disk_write(nbytes)
            return self._maybe_pin(st)

        with timed(self.stats, "disk_read"):
            st = self.disk.load(expert_id)
        nbytes = state_nbytes(st)
        self.stats.add_bytes("disk_read_bytes", nbytes)
        self.stats.inc("disk_read_ops", 1)
        self.lat.simulate_disk_read(nbytes)
        return self._maybe_pin(st)

    def _cpu_get(self, expert_id: int) -> CPUResident:
        """Return CPU-resident canonical state, loading from disk if needed."""

        # Drain prefetch completions first so we benefit from background reads.
        self.drain_prefetch()

        entry = self.cpu_cache.get(expert_id)
        if entry is not None:
            self.stats.inc("cpu_cache_hit", 1)
            return entry

        self.stats.inc("cpu_cache_miss", 1)

        if self.disk is None:
            # No cold store; lazily init and keep in warm cache if possible.
            st = make_expert_state(
                self.d_model,
                self.d_hidden,
                seed=self.seed_base + expert_id,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.pin_cpu,
                with_momentum=False,
            )
            entry = CPUResident(state=st, dirty=True)
            if self.cpu_cache.capacity > 0:
                evicted, _ = self.cpu_cache.put(expert_id, entry)
                if evicted is not None:
                    self.stats.inc("cpu_evictions_no_disk", 1)
            return entry

        # If a prefetch is in-flight, wait for it.
        fut = self._prefetch_futs.pop(expert_id, None)
        if fut is not None:
            self.stats.inc("prefetch_waited", 1)
            st = fut.result()
        else:
            st = self._disk_load_or_init_sync(expert_id)

        entry = CPUResident(state=st, dirty=False)
        if self.cpu_cache.capacity > 0:
            evicted, _ = self.cpu_cache.put(expert_id, entry)
            if evicted is not None:
                old_id, old_entry = evicted
                self._on_cpu_evict(old_id, old_entry)
        return entry

    def _disk_save_worker(self, expert_id: int, st: ExpertState) -> None:
        assert self.disk is not None
        with timed(self.stats, "disk_write"):
            self.disk.save(expert_id, st)
        nbytes = state_nbytes(st)
        self.stats.add_bytes("disk_write_bytes", nbytes)
        self.stats.inc("disk_write_ops", 1)
        self.lat.simulate_disk_write(nbytes)

    def _on_cpu_evict(self, expert_id: int, entry: CPUResident) -> None:
        self.stats.inc("cpu_evictions", 1)
        if not entry.dirty:
            self.stats.inc("cpu_evict_clean", 1)
            return
        if self.disk is None:
            self.stats.inc("cpu_evict_dirty_no_disk", 1)
            return

        # Safe to write asynchronously: the entry is no longer referenced.
        if self._io_executor is None:
            self._disk_save_worker(expert_id, entry.state)
            return

        self.stats.inc("writeback_scheduled", 1)
        fut = self._io_executor.submit(self._disk_save_worker, int(expert_id), entry.state)
        self._writeback_futs.append(fut)

    # ----------------------------
    # Prefetch API
    # ----------------------------

    def prefetch(self, expert_ids: Iterable[int]) -> None:
        """Asynchronously prefetch experts from disk into a ready queue.

        This only performs disk->CPU reads. The main thread must call drain_prefetch()
        periodically to insert ready states into the warm CPU cache.
        """
        if self.disk is None or self._io_executor is None:
            return

        # Best-effort; if the expert is already in CPU cache or GPU cache, skip.
        for eid in set(int(e) for e in expert_ids):
            if eid in self.gpu_map:
                continue
            if eid in self.cpu_cache:
                # Don't perturb LRU just to check presence.
                self.stats.inc("prefetch_hit_cpu", 1)
                continue
            if eid in self._prefetch_futs:
                continue

            self.stats.inc("prefetch_scheduled", 1)
            fut = self._io_executor.submit(self._disk_load_or_init_worker, eid)
            self._prefetch_futs[eid] = fut

            def _cb(f: Future, _eid: int = eid) -> None:
                try:
                    st = f.result()
                except Exception:
                    self.stats.inc("prefetch_failed", 1)
                    return
                self._prefetch_ready.put((_eid, st))
                self.stats.inc("prefetch_completed", 1)

            fut.add_done_callback(_cb)

    def drain_prefetch(self, max_items: int = 32) -> int:
        """Move completed prefetch reads into the CPU warm cache."""
        moved = 0
        if self.disk is None:
            return 0
        while moved < max_items:
            try:
                eid, st = self._prefetch_ready.get_nowait()  # type: ignore[attr-defined]
            except Exception:
                break

            # Remove from in-flight dict if still present
            self._prefetch_futs.pop(eid, None)

            if self.cpu_cache.capacity == 0:
                # No warm cache => nothing to do.
                moved += 1
                continue

            evicted, _ = self.cpu_cache.put(int(eid), CPUResident(state=st, dirty=False))
            if evicted is not None:
                old_id, old_entry = evicted
                self._on_cpu_evict(old_id, old_entry)
            moved += 1
        return moved

    # ----------------------------
    # GPU residency
    # ----------------------------

    def ensure_on_gpu(self, expert_id: int) -> Tuple[ExpertSlot, int]:
        """Ensure expert is resident in a GPU slot. Returns (slot, slot_idx)."""
        res = self.gpu_map.get(expert_id)
        if res is not None:
            self.stats.inc("gpu_cache_hit", 1)
            return self.slots[res.slot_idx], res.slot_idx
        self.stats.inc("gpu_cache_miss", 1)

        cpu_entry = self._cpu_get(expert_id)

        if self.free_slots:
            slot_idx = self.free_slots.pop()
        else:
            old_eid = next(iter(self.gpu_map.keys()))
            old_res = self.gpu_map.pop(old_eid)
            assert old_res is not None
            self.stats.inc("gpu_evictions", 1)
            slot_idx = old_res.slot_idx

        slot = self.slots[slot_idx]
        with timed(self.stats, "h2d_load"):
            moved = slot.load_from_cpu(cpu_entry.state, non_blocking=True)
        self.stats.add_bytes("h2d_bytes", moved)
        self.stats.inc("h2d_ops", 1)
        self.lat.simulate_h2d(moved)

        self.gpu_map.put(expert_id, GPUResident(slot_idx=slot_idx))
        return slot, slot_idx

    # ----------------------------
    # Optimizer update
    # ----------------------------

    def _ensure_opt(self, st: ExpertState) -> None:
        if self.optim == "sgd":
            st.opt = None
            return
        if st.opt is not None and st.opt.kind == self.optim:
            return

        tensors: TensorDict = {}
        if self.optim == "sgd_momentum":
            for k, p in st.params.items():
                tensors[f"m_{k}"] = torch.zeros_like(p)
        elif self.optim == "adamw":
            for k, p in st.params.items():
                tensors[f"m_{k}"] = torch.zeros_like(p)
                tensors[f"v_{k}"] = torch.zeros_like(p)
        else:
            raise ValueError(f"Unknown optim: {self.optim}")

        if self.pin_cpu and torch.cuda.is_available():
            for k in list(tensors.keys()):
                if not tensors[k].is_pinned():
                    tensors[k] = tensors[k].pin_memory()

        st.opt = ExpertOptState(kind=self.optim, step=0, tensors=tensors)

    @torch.no_grad()
    def step_expert(self, expert_id: int, *, lr: float, weight_decay: float = 0.0) -> None:
        """Apply an optimizer step for a GPU-resident expert.

        - Copies grads GPU->CPU
        - Updates CPU master weights + optimizer state
        - Optionally writes through / periodic flush
        - Copies updated CPU weights back to GPU slot (so caching remains correct)
        """
        res = self.gpu_map.get(expert_id)
        if res is None:
            raise RuntimeError("Expert not on GPU; call ensure_on_gpu first.")

        slot = self.slots[res.slot_idx]

        # 1) GPU -> CPU grads
        with timed(self.stats, "d2h_grads"):
            grads = slot.grads_to_cpu()
        nbytes = sum(tensor_nbytes(g) for g in grads.values())
        self.stats.add_bytes("d2h_bytes", nbytes)
        self.stats.inc("d2h_ops", 1)
        self.lat.simulate_d2h(nbytes)

        # 2) CPU update
        cpu_entry = self._cpu_get(expert_id)
        st = cpu_entry.state
        self._ensure_opt(st)

        if self.optim == "sgd":
            for k, p in st.params.items():
                g = grads[k]
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)
                p.add_(g, alpha=-lr)
        elif self.optim == "sgd_momentum":
            assert st.opt is not None
            mu = self.momentum
            for k, p in st.params.items():
                g = grads[k]
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)
                m = st.opt.tensors[f"m_{k}"]
                m.mul_(mu).add_(g, alpha=1.0)
                p.add_(m, alpha=-lr)
        elif self.optim == "adamw":
            assert st.opt is not None
            st.opt.step += 1
            t = st.opt.step
            b1, b2 = self.adam_beta1, self.adam_beta2
            eps = self.adam_eps
            # Decoupled weight decay
            if weight_decay != 0.0:
                for _, p in st.params.items():
                    p.mul_(1.0 - lr * weight_decay)
            bias_c1 = 1.0 - (b1**t)
            bias_c2 = 1.0 - (b2**t)
            for k, p in st.params.items():
                g = grads[k]
                m = st.opt.tensors[f"m_{k}"]
                v = st.opt.tensors[f"v_{k}"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                m_hat = m / bias_c1
                v_hat = v / bias_c2
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        else:
            raise ValueError(f"Unknown optim: {self.optim}")

        cpu_entry.dirty = True

        # update LRU position
        if self.cpu_cache.capacity > 0:
            self.cpu_cache.put(expert_id, cpu_entry)

        # Optional synchronous flushing
        self._step_counter += 1
        if self.disk is not None:
            if self.writeback_policy == "writethrough":
                # Safe: synchronous, and immediately marks clean.
                self._disk_save_worker(expert_id, st)
                cpu_entry.dirty = False
            elif self.writeback_policy == "periodic" and self.writeback_every > 0:
                if self._step_counter % self.writeback_every == 0:
                    self.flush_dirty(sync=True)

        # 3) CPU -> GPU: keep slot weights consistent for caching
        with timed(self.stats, "h2d_sync"):
            moved = slot.load_from_cpu(st, non_blocking=True)
        self.stats.add_bytes("h2d_bytes", moved)
        self.stats.inc("h2d_ops", 1)
        self.lat.simulate_h2d(moved)

        slot.zero_grad()
        self.gpu_map.put(expert_id, res)

    # ----------------------------
    # Flushing / shutdown
    # ----------------------------

    def flush_dirty(self, *, sync: bool = True) -> None:
        """Flush dirty CPU-resident experts to disk.

        If sync=True, performs writes in the caller thread (safe if entries remain resident).
        If sync=False, schedules async writes and marks entries clean immediately.
        """
        if self.disk is None:
            return

        for eid, entry in list(self.cpu_cache.items()):
            if not entry.dirty:
                continue
            if sync:
                self._disk_save_worker(int(eid), entry.state)
                entry.dirty = False
                self.cpu_cache.put(int(eid), entry)
            else:
                # Not strictly safe if the entry could be modified before write completes.
                # Provided for experimentation only.
                if self._io_executor is None:
                    self._disk_save_worker(int(eid), entry.state)
                else:
                    fut = self._io_executor.submit(self._disk_save_worker, int(eid), entry.state)
                    self._writeback_futs.append(fut)
                entry.dirty = False

    def flush_all(self) -> None:
        """Flush everything and shut down background I/O."""
        # Evict everything from GPU map
        for _, res in list(self.gpu_map.items()):
            self.free_slots.add(res.slot_idx)
        self.gpu_map = LRUCache(self.gpu_map.capacity)

        # Sync flush dirty CPU entries
        self.flush_dirty(sync=True)

        # Wait for async writebacks to complete
        for fut in list(self._writeback_futs):
            try:
                fut.result()
            except Exception:
                self.stats.inc("writeback_failed", 1)
        self._writeback_futs.clear()

        # Shut down executor
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=True)
            self._io_executor = None

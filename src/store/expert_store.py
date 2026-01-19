from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class ExpertStoreStats:
    # Access stats
    hbm_hits: int = 0
    hbm_misses: int = 0
    # IO stats
    nvme_reads: int = 0
    nvme_writes: int = 0
    bytes_nvme_read: int = 0
    bytes_nvme_write: int = 0
    bytes_h2d: int = 0
    bytes_d2h: int = 0
    # Evictions
    hbm_evictions: int = 0
    dram_evictions: int = 0
    # Timing
    stall_time_s: float = 0.0

    def hit_rate(self) -> float:
        denom = self.hbm_hits + self.hbm_misses
        return float(self.hbm_hits) / float(denom) if denom > 0 else 0.0


def _state_dict_to_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _optimizer_state_to_cpu(state: Dict) -> Dict:
    # Deep-ish copy where tensor leaves are moved to CPU.
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        elif isinstance(v, dict):
            out[k] = _optimizer_state_to_cpu(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [(_optimizer_state_to_cpu(x) if isinstance(x, dict) else (x.detach().cpu() if torch.is_tensor(x) else x)) for x in v]
        else:
            out[k] = v
    return out


def _move_optimizer_state_(optim: torch.optim.Optimizer, device: torch.device) -> None:
    # In-place move of optimizer state tensors.
    for state in optim.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _module_device(module: nn.Module) -> torch.device:
    """Best-effort device of a module (first parameter device, else CPU)."""
    p = next(module.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


class ExpertStore:
    """A minimal "real" expert store that pages experts across GPU (HBM), CPU (DRAM), and disk (NVMe).

    This is intentionally a research scaffold:
    - Experts are nn.Modules already owned by a parent model (typically a ModuleList).
    - The store controls where each expert's parameters live.
    - It maintains simple LRU caches for GPU and CPU tiers.
    - When evicting from CPU beyond budget, it writes weights (+ optional optimizer state) to disk and moves
      the expert module to the 'meta' device to free RAM.

    Notes:
    - Disk tier uses torch.save/torch.load. It's not the fastest, but it's simple and robust.
    - This store is designed to support sparse MoE usage (top-k experts per step).
    """

    def __init__(
        self,
        *,
        experts: nn.ModuleList,
        disk_dir: str | Path,
        hbm_capacity: int,
        dram_capacity: int,
        device: torch.device,
        enable_disk: bool = True,
        pin_cpu: bool = True,
        create_expert_optimizer: bool = True,
        expert_lr: float = 1e-4,
    ):
        self.experts = experts
        self.num_experts = len(experts)
        self.device = device
        self.hbm_capacity = int(hbm_capacity)
        self.dram_capacity = int(dram_capacity)
        self.enable_disk = bool(enable_disk)
        self.pin_cpu = bool(pin_cpu)
        # Pinning only makes sense when CUDA is present; CPU-only builds may error on pin_memory().
        self._can_pin = bool(self.pin_cpu and self.device.type == "cuda" and torch.cuda.is_available())
        self.create_expert_optimizer = bool(create_expert_optimizer)
        self.expert_lr = float(expert_lr)

        self.disk_dir = Path(disk_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        # LRU caches implemented as ordered lists (MRU at front)
        self._hbm_lru: List[int] = []
        self._dram_lru: List[int] = []

        # Per-expert optimizer (only for the online network). Target store can disable.
        self._optims: Dict[int, torch.optim.Optimizer] = {}
        self._optim_state_disk: Dict[int, Path] = {}

        self.stats = ExpertStoreStats()

    # ------------------ paths ------------------

    def expert_path(self, expert_id: int) -> Path:
        return self.disk_dir / f"expert_{int(expert_id):05d}.pt"

    # ------------------ stats ------------------

    def reset_stats(self) -> ExpertStoreStats:
        old = self.stats
        self.stats = ExpertStoreStats()
        return old

    # ------------------ residency helpers ------------------

    def _is_on_gpu(self, expert_id: int) -> bool:
        e = self.experts[int(expert_id)]
        p = next(e.parameters(), None)
        if p is None:
            return False
        if p.device.type != self.device.type:
            return False

        # torch.device('cuda') is not equal to torch.device('cuda:0'). If the store
        # was initialized with an index-less CUDA device, treat any CUDA index as GPU-resident.
        if self.device.type == "cuda" and self.device.index is None:
            return True

        return p.device == self.device

    def _is_on_cpu(self, expert_id: int) -> bool:
        e = self.experts[int(expert_id)]
        p = next(e.parameters(), None)
        if p is None:
            return False
        return p.device.type == "cpu"

    def _is_on_meta(self, expert_id: int) -> bool:
        e = self.experts[int(expert_id)]
        p = next(e.parameters(), None)
        if p is None:
            return False
        return p.device.type == "meta"

    def _touch_lru(self, lru: List[int], expert_id: int) -> None:
        expert_id = int(expert_id)
        if expert_id in lru:
            lru.remove(expert_id)
        lru.insert(0, expert_id)

    def _remove_lru(self, lru: List[int], expert_id: int) -> None:
        expert_id = int(expert_id)
        if expert_id in lru:
            lru.remove(expert_id)

    # ------------------ disk IO ------------------

    def _save_to_disk(self, expert_id: int) -> None:
        if not self.enable_disk:
            return

        expert_id = int(expert_id)
        e = self.experts[expert_id]
        path = self.expert_path(expert_id)

        # Ensure weights are on CPU for portability
        sd_cpu = _state_dict_to_cpu(e.state_dict())

        payload = {"state_dict": sd_cpu}

        if self.create_expert_optimizer and expert_id in self._optims:
            opt = self._optims[expert_id]
            opt_sd = opt.state_dict()
            payload["optim_state"] = _optimizer_state_to_cpu(opt_sd)

        t0 = time.perf_counter()
        torch.save(payload, path)
        dt = time.perf_counter() - t0

        # Approx bytes written
        try:
            nbytes = path.stat().st_size
        except Exception:
            nbytes = 0

        self.stats.nvme_writes += 1
        self.stats.bytes_nvme_write += int(nbytes)
        self.stats.stall_time_s += float(dt)

    def _load_from_disk_to_cpu(self, expert_id: int) -> None:
        expert_id = int(expert_id)
        path = self.expert_path(expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert {expert_id} not found on disk at {path}")

        t0 = time.perf_counter()
        # NOTE: We avoid using newer torch.load(...) keyword args (e.g. weights_only)
        # to stay compatible with the wide range of PyTorch versions seen in Colab.
        payload = torch.load(path, map_location="cpu")
        dt = time.perf_counter() - t0

        try:
            nbytes = path.stat().st_size
        except Exception:
            nbytes = 0

        self.stats.nvme_reads += 1
        self.stats.bytes_nvme_read += int(nbytes)
        self.stats.stall_time_s += float(dt)

        e = self.experts[expert_id]
        # If expert is meta, materialize empty CPU tensors first.
        if self._is_on_meta(expert_id):
            e = e.to_empty(device="cpu")
        else:
            e = e.to("cpu")

        e.load_state_dict(payload["state_dict"], strict=True)

        if self._can_pin:
            for p in e.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.pin_memory()

        # Restore / create optimizer state
        if self.create_expert_optimizer:
            # (re)create optimizer if missing
            if expert_id not in self._optims:
                self._optims[expert_id] = torch.optim.Adam(e.parameters(), lr=self.expert_lr)

            opt = self._optims[expert_id]
            if "optim_state" in payload:
                opt.load_state_dict(payload["optim_state"])

    # ------------------ tier transitions ------------------

    def _ensure_on_cpu(self, expert_id: int) -> None:
        expert_id = int(expert_id)
        if self._is_on_cpu(expert_id):
            # Even if the module is already on CPU, make sure (a) the per-expert
            # optimizer exists if requested and (b) its state tensors are on CPU.
            if self.create_expert_optimizer:
                if expert_id not in self._optims:
                    self._optims[expert_id] = torch.optim.Adam(self.experts[expert_id].parameters(), lr=self.expert_lr)
                _move_optimizer_state_(self._optims[expert_id], torch.device("cpu"))

            self._touch_lru(self._dram_lru, expert_id)
            self._evict_dram_if_needed()
            return

        # If on disk/meta, load from disk. Otherwise move from GPU.
        if self._is_on_meta(expert_id):
            self._load_from_disk_to_cpu(expert_id)
        else:
            # From GPU -> CPU
            e = self.experts[expert_id]
            # Bookkeeping: no longer GPU-resident.
            self._remove_lru(self._hbm_lru, expert_id)
            # Approx d2h bytes
            d2h = sum(int(p.numel() * p.element_size()) for p in e.parameters())
            t0 = time.perf_counter()
            e = e.to("cpu")
            dt = time.perf_counter() - t0
            self.stats.bytes_d2h += int(d2h)
            self.stats.stall_time_s += float(dt)

            if self._can_pin:
                for p in e.parameters():
                    p.data = p.data.pin_memory()

            # Move optimizer state to CPU
            if self.create_expert_optimizer and expert_id in self._optims:
                _move_optimizer_state_(self._optims[expert_id], torch.device("cpu"))

        self._touch_lru(self._dram_lru, expert_id)
        self._evict_dram_if_needed()

    def _ensure_on_gpu(self, expert_id: int) -> None:
        expert_id = int(expert_id)
        if self._is_on_gpu(expert_id):
            # IMPORTANT: parameters can end up on GPU while optimizer state is still
            # on CPU (e.g., after a checkpoint restore, or accidental external `.to()` calls).
            # Make this path robust by ensuring optimizer/state residency too.
            if self.create_expert_optimizer:
                if expert_id not in self._optims:
                    self._optims[expert_id] = torch.optim.Adam(self.experts[expert_id].parameters(), lr=self.expert_lr)
                _move_optimizer_state_(self._optims[expert_id], self.device)

            self._touch_lru(self._hbm_lru, expert_id)
            self.stats.hbm_hits += 1
            return

        self.stats.hbm_misses += 1

        # Ensure at least CPU resident first
        self._ensure_on_cpu(expert_id)

        e = self.experts[expert_id]
        # Approx h2d bytes
        h2d = sum(int(p.numel() * p.element_size()) for p in e.parameters())

        t0 = time.perf_counter()
        e = e.to(self.device, non_blocking=True)
        # If we want true overlap we'd need streams; here we time the blocking move.
        if self.device.type == "cuda":
            # Synchronize the current CUDA device for timing stability.
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        self.stats.bytes_h2d += int(h2d)
        self.stats.stall_time_s += float(dt)

        # Bookkeeping: after moving to GPU, it is no longer in DRAM.
        self._remove_lru(self._dram_lru, expert_id)

        # Move optimizer state to GPU
        if self.create_expert_optimizer:
            if expert_id not in self._optims:
                # If we didn't have it (e.g. expert was on disk without optimizer yet)
                self._optims[expert_id] = torch.optim.Adam(e.parameters(), lr=self.expert_lr)
            _move_optimizer_state_(self._optims[expert_id], self.device)

        self._touch_lru(self._hbm_lru, expert_id)
        self._evict_hbm_if_needed(protect=set([expert_id]))

    def _evict_hbm_if_needed(self, protect: Optional[Set[int]] = None) -> None:
        protect = protect or set()
        while len(self._hbm_lru) > self.hbm_capacity:
            # Evict least recently used that is not protected.
            evict_id = None
            for cand in reversed(self._hbm_lru):
                if cand not in protect:
                    evict_id = cand
                    break
            if evict_id is None:
                # If everything is protected, evict the absolute LRU.
                evict_id = self._hbm_lru[-1]

            self._hbm_lru.remove(evict_id)
            self.stats.hbm_evictions += 1
            # Move to CPU (DRAM)
            self._ensure_on_cpu(evict_id)

    def _evict_dram_if_needed(self) -> None:
        if not self.enable_disk:
            # Without disk tier, just keep on CPU; ignore DRAM budget.
            return

        while len(self._dram_lru) > self.dram_capacity:
            evict_id = self._dram_lru.pop()  # LRU
            self.stats.dram_evictions += 1

            # Persist to disk
            self._save_to_disk(evict_id)

            # Drop optimizer (will be recreated on reload)
            if evict_id in self._optims:
                del self._optims[evict_id]

            # Move module to meta to free CPU memory
            e = self.experts[int(evict_id)]
            e.to("meta")

    # ------------------ public API ------------------

    def cold_start_to_disk(self) -> None:
        """Save all experts to disk and move them to meta (cold start)."""
        if not self.enable_disk:
            return

        for eid in range(self.num_experts):
            # Save (weights + any optimizer state if present)
            self._save_to_disk(eid)
            # Drop optimizer
            if eid in self._optims:
                del self._optims[eid]
            # Move to meta
            self.experts[eid].to("meta")

        self._hbm_lru.clear()
        self._dram_lru.clear()

    def prefetch_to_gpu(self, expert_ids: Sequence[int]) -> None:
        # Best-effort: ensure these experts on GPU now.
        uniq: List[int] = []
        for e in expert_ids:
            e = int(e)
            if e not in uniq:
                uniq.append(e)
        protect = set(uniq)
        for e in uniq:
            self._ensure_on_gpu(e)
        # enforce cap with protection (try not to evict prefetched)
        self._evict_hbm_if_needed(protect=protect)

    def ensure_on_gpu(self, expert_ids: Sequence[int]) -> None:
        uniq: List[int] = []
        for e in expert_ids:
            e = int(e)
            if e not in uniq:
                uniq.append(e)
        protect = set(uniq)
        for e in uniq:
            self._ensure_on_gpu(e)
        self._evict_hbm_if_needed(protect=protect)

    def ensure_on_cpu(self, expert_ids: Sequence[int]) -> None:
        """Ensure experts are materialized on CPU (DRAM), without moving them to GPU."""
        uniq: List[int] = []
        for e in expert_ids:
            e = int(e)
            if e not in uniq:
                uniq.append(e)
        for e in uniq:
            self._ensure_on_cpu(e)

    def prefetch_to_cpu(self, expert_ids: Sequence[int]) -> None:
        """Alias for ensure_on_cpu; included for symmetry."""
        self.ensure_on_cpu(expert_ids)

    def get_optimizer(self, expert_id: int) -> Optional[torch.optim.Optimizer]:
        return self._optims.get(int(expert_id))

    def zero_grad(self, expert_ids: Iterable[int]) -> None:
        for e in set(int(x) for x in expert_ids):
            opt = self._optims.get(e)
            if opt is not None:
                opt.zero_grad(set_to_none=True)

    def step(self, expert_ids: Iterable[int]) -> None:
        # Defensive: experts may be moved across devices (GPU<->CPU) between the
        # backward pass and the optimizer step (e.g., due to paging / eviction).
        #
        # PyTorch optimizers do NOT automatically move their state tensors, and
        # module.to(device) does not reliably move parameter .grad tensors.
        #
        # To prevent "cuda vs cpu" errors inside Adam, we align BOTH:
        #   (1) optimizer state tensors (exp_avg, exp_avg_sq, ...) and
        #   (2) parameter gradients
        # to the expert's current parameter device before calling opt.step().
        for e in set(int(x) for x in expert_ids):
            opt = self._optims.get(e)
            if opt is None:
                continue
            mod = self.experts[e]
            dev = _module_device(mod)

            # Ensure gradients are on the same device as parameters.
            for p in mod.parameters():
                if p.grad is not None and p.grad.device != dev:
                    p.grad = p.grad.to(dev, non_blocking=True)

            _move_optimizer_state_(opt, dev)
            opt.step()

    def experts_on_gpu(self) -> List[int]:
        return list(self._hbm_lru)

    def experts_on_cpu(self) -> List[int]:
        return list(self._dram_lru)

    def flush_resident_to_disk(self) -> None:
        """Best-effort: persist all currently materialized experts to disk.

        This is useful in Colab so that you don't lose expert weights/optim state if the runtime dies.
        """
        if not self.enable_disk:
            return
        ids = set(self._hbm_lru + self._dram_lru)
        for eid in sorted(ids):
            self._save_to_disk(eid)

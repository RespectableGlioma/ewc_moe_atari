from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

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

    def miss_rate(self) -> float:
        denom = self.hbm_hits + self.hbm_misses
        return float(self.hbm_misses) / float(denom) if denom > 0 else 0.0


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
    out: Dict = {}
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
    """Pages experts across GPU (HBM), CPU (DRAM), and disk (NVMe).

    Key correctness feature:
      - Step-scoped pinning via begin_step()/end_step(): experts that participate in the current
        autograd graph are protected from eviction until end_step().

    This is a research scaffold and intentionally simple:
      - Disk tier uses torch.save/torch.load.
      - GPU/CPU tiers use a basic LRU.
      - Online store optionally maintains per-expert Adam optimizers.
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

        # Step-scoped pinning: prevent eviction of experts participating in the current autograd graph.
        self._pin_active: bool = False
        self._pinned_hbm: Set[int] = set()

        self.disk_dir = Path(disk_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        # LRU caches implemented as ordered lists (MRU at front)
        self._hbm_lru: List[int] = []
        self._dram_lru: List[int] = []

        # Per-expert optimizer (only for the online network). Target store can disable.
        self._optims: Dict[int, torch.optim.Optimizer] = {}

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

        # torch.device('cuda') != torch.device('cuda:0'); treat any cuda index as resident when store device has no index.
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
        # Meta device means "no storage" (only shapes). You cannot serialize parameters from
        # a meta expert because there is literally no data to write. This can happen if
        # bookkeeping is stale (an expert ID is still in an LRU list after being dropped),
        # or if an expert was intentionally offloaded to meta earlier.
        #
        # If a disk copy already exists, skipping is safe: the meta expert cannot contain
        # any newer weights than what's on disk.
        if self._is_on_meta(expert_id):
            if path.exists():
                return
            raise RuntimeError(
                f"Cannot save expert {expert_id}: expert is on meta device and no disk copy exists at {path}. "
                "This indicates a paging/bookkeeping bug (attempted to persist an expert with no materialized weights)."
            )


        payload: Dict = {"state_dict": _state_dict_to_cpu(e.state_dict())}

        if self.create_expert_optimizer and expert_id in self._optims:
            opt_sd = self._optims[expert_id].state_dict()
            payload["optim_state"] = _optimizer_state_to_cpu(opt_sd)

        t0 = time.perf_counter()
        torch.save(payload, path)
        dt = time.perf_counter() - t0

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
        if self._is_on_meta(expert_id):
            e = e.to_empty(device="cpu")
        else:
            e = e.to("cpu")

        e.load_state_dict(payload["state_dict"], strict=True)

        if self._can_pin:
            for p in e.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.pin_memory()

        if self.create_expert_optimizer:
            if expert_id not in self._optims:
                self._optims[expert_id] = torch.optim.Adam(e.parameters(), lr=self.expert_lr)
            opt = self._optims[expert_id]
            if "optim_state" in payload:
                opt.load_state_dict(payload["optim_state"])

    # ------------------ step-scoped pinning ------------------

    def begin_step(self) -> None:
        """Call before a training forward that will be followed by backward()."""
        self._pin_active = True
        self._pinned_hbm.clear()

    def end_step(self) -> None:
        """Call after optimizer steps; now it's safe to evict again."""
        self._pin_active = False
        self._pinned_hbm.clear()
        # Enforce budget after the step (evictions are safe now).
        self._evict_hbm_if_needed(protect=set())

    # ------------------ tier transitions ------------------

    def _ensure_on_cpu(self, expert_id: int) -> None:
        expert_id = int(expert_id)

        if self._is_on_cpu(expert_id):
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
            e = self.experts[expert_id]
            self._remove_lru(self._hbm_lru, expert_id)

            d2h = sum(int(p.numel() * p.element_size()) for p in e.parameters())
            t0 = time.perf_counter()
            e = e.to("cpu")
            dt = time.perf_counter() - t0

            self.stats.bytes_d2h += int(d2h)
            self.stats.stall_time_s += float(dt)

            if self._can_pin:
                for p in e.parameters():
                    p.data = p.data.pin_memory()

            if self.create_expert_optimizer and expert_id in self._optims:
                _move_optimizer_state_(self._optims[expert_id], torch.device("cpu"))

        self._touch_lru(self._dram_lru, expert_id)
        self._evict_dram_if_needed()

    def _ensure_on_gpu(self, expert_id: int) -> None:
        expert_id = int(expert_id)

        if self._is_on_gpu(expert_id):
            # Ensure optimizer state lives on the same device as params.
            if self.create_expert_optimizer:
                if expert_id not in self._optims:
                    self._optims[expert_id] = torch.optim.Adam(self.experts[expert_id].parameters(), lr=self.expert_lr)
                _move_optimizer_state_(self._optims[expert_id], self.device)

            self._touch_lru(self._hbm_lru, expert_id)
            if self._pin_active:
                self._pinned_hbm.add(expert_id)
            self.stats.hbm_hits += 1
            return

        self.stats.hbm_misses += 1

        # Ensure at least CPU resident first
        self._ensure_on_cpu(expert_id)

        e = self.experts[expert_id]
        h2d = sum(int(p.numel() * p.element_size()) for p in e.parameters())

        t0 = time.perf_counter()
        e = e.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        self.stats.bytes_h2d += int(h2d)
        self.stats.stall_time_s += float(dt)

        # After moving to GPU, it is no longer DRAM-resident.
        self._remove_lru(self._dram_lru, expert_id)

        # Move optimizer state to GPU
        if self.create_expert_optimizer:
            if expert_id not in self._optims:
                self._optims[expert_id] = torch.optim.Adam(e.parameters(), lr=self.expert_lr)
            _move_optimizer_state_(self._optims[expert_id], self.device)

        self._touch_lru(self._hbm_lru, expert_id)
        if self._pin_active:
            self._pinned_hbm.add(expert_id)

        # Enforce cap (but do not evict the one we just brought in).
        self._evict_hbm_if_needed(protect=set([expert_id]))

    def _evict_hbm_if_needed(self, protect: Optional[Set[int]] = None) -> None:
        protect = protect or set()

        # Always protect pinned experts during an active autograd step.
        if self._pin_active and self._pinned_hbm:
            protect |= set(self._pinned_hbm)

        while len(self._hbm_lru) > self.hbm_capacity:
            # Evict LRU that is not protected.
            evict_id: Optional[int] = None
            for cand in reversed(self._hbm_lru):
                if cand not in protect:
                    evict_id = cand
                    break

            if evict_id is None:
                # Everything currently resident is protected.
                # If we're inside an active step, it's unsafe to evict (would break autograd).
                if self._pin_active:
                    break
                # Outside a step, evict absolute LRU.
                evict_id = self._hbm_lru[-1]

            self._hbm_lru.remove(int(evict_id))
            self.stats.hbm_evictions += 1
            self._ensure_on_cpu(int(evict_id))

    def _evict_dram_if_needed(self) -> None:
        if not self.enable_disk:
            return

        while len(self._dram_lru) > self.dram_capacity:
            evict_id = int(self._dram_lru.pop())
            self.stats.dram_evictions += 1

            self._save_to_disk(evict_id)

            if evict_id in self._optims:
                del self._optims[evict_id]

            self.experts[evict_id].to("meta")

    # ------------------ public API ------------------

    def cold_start_to_disk(self) -> None:
        """Save all experts to disk and move them to meta (cold start)."""
        if not self.enable_disk:
            return

        for eid in range(self.num_experts):
            self._save_to_disk(eid)
            if eid in self._optims:
                del self._optims[eid]
            self.experts[eid].to("meta")

        self._hbm_lru.clear()
        self._dram_lru.clear()

    def prefetch_to_gpu(self, expert_ids: Sequence[int]) -> None:
        uniq: List[int] = []
        for e in expert_ids:
            e = int(e)
            if e not in uniq:
                uniq.append(e)
        protect = set(uniq)
        for e in uniq:
            self._ensure_on_gpu(e)
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
        uniq: List[int] = []
        for e in expert_ids:
            e = int(e)
            if e not in uniq:
                uniq.append(e)
        for e in uniq:
            self._ensure_on_cpu(e)

    def prefetch_to_cpu(self, expert_ids: Sequence[int]) -> None:
        self.ensure_on_cpu(expert_ids)

    def get_optimizer(self, expert_id: int) -> Optional[torch.optim.Optimizer]:
        return self._optims.get(int(expert_id))

    def zero_grad(self, expert_ids: Iterable[int]) -> None:
        for e in set(int(x) for x in expert_ids):
            opt = self._optims.get(e)
            if opt is not None:
                opt.zero_grad(set_to_none=True)

    def step(self, expert_ids: Iterable[int]) -> None:
        # Align parameter grads and optimizer state device before stepping.
        for e in set(int(x) for x in expert_ids):
            opt = self._optims.get(e)
            if opt is None:
                continue
            mod = self.experts[e]
            dev = _module_device(mod)

            for p in mod.parameters():
                if p.grad is not None and p.grad.device != dev:
                    p.grad = p.grad.to(dev, non_blocking=True)

            _move_optimizer_state_(opt, dev)
            opt.step()

    def experts_on_gpu(self) -> List[int]:
        return list(self._hbm_lru)

    def experts_on_cpu(self) -> List[int]:
        return list(self._dram_lru)


    def reinit_expert(
        self,
        expert_id: int,
        *,
        seed: Optional[int] = None,
        save_to_disk: bool = False,
        move_to: str = "cpu",
    ) -> None:
        """Re-initialize an expert's weights (and optimizer state).

        This is used by "spawn a new expert" style algorithms: you want a fresh expert slot
        without needing to grow `num_experts`.

        Notes:
        - If the expert currently lives on the meta device, we materialize it via `to_empty()`
          on the init device before resetting parameters.
        - If `save_to_disk=True`, we persist the fresh weights/optim state immediately so
          later drops to meta are safe.
        """
        expert_id = int(expert_id)

        # Deterministic-ish init when desired.
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Where do we want the expert to live after reinit?
        move_to = str(move_to).lower()
        if move_to in ("gpu", "cuda"):
            target = self.device
        elif move_to in ("cpu",):
            target = torch.device("cpu")
        elif move_to in ("meta",):
            target = torch.device("meta")
        else:
            # Allow a fully-qualified device string (e.g., "cuda:0"), else fall back to CPU.
            try:
                target = torch.device(move_to)
            except Exception:
                target = torch.device("cpu")

        # You cannot initialize parameters on meta; materialize on CPU first, then (optionally) move.
        init_dev = torch.device("cpu") if target.type == "meta" else target

        # Remove from LRUs (we will re-touch below).
        self._remove_lru(self._hbm_lru, expert_id)
        self._remove_lru(self._dram_lru, expert_id)

        e = self.experts[expert_id]

        # Materialize on init_dev.
        if init_dev.type == "cpu":
            if self._is_on_meta(expert_id):
                e = e.to_empty(device="cpu")
            else:
                e = e.to("cpu")
        else:
            if self._is_on_meta(expert_id):
                e = e.to_empty(device=init_dev)
            else:
                e = e.to(init_dev)

        # Reset parameters for all submodules that support it.
        def _reset(m: nn.Module) -> None:
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()
                except Exception:
                    pass

        with torch.no_grad():
            e.apply(_reset)

        # Replace the module reference (defensive; .to() returns the same object in-place, but keep it explicit).
        self.experts[expert_id] = e

        # (Re)create optimizer state for this expert.
        if expert_id in self._optims:
            del self._optims[expert_id]
        if self.create_expert_optimizer:
            self._optims[expert_id] = torch.optim.Adam(e.parameters(), lr=self.expert_lr)
            _move_optimizer_state_(self._optims[expert_id], init_dev)

        # Pin CPU tensors if requested.
        if init_dev.type == "cpu" and self._can_pin:
            for p in e.parameters():
                if p.device.type == "cpu":
                    p.data = p.data.pin_memory()

        # Optionally persist immediately (so later drops-to-meta are safe).
        if save_to_disk and self.enable_disk:
            self._save_to_disk(expert_id)

        # Move to final tier if needed.
        if target.type == "meta":
            # Drop to meta (free memory). Optim state is dropped too.
            self._remove_lru(self._hbm_lru, expert_id)
            self._remove_lru(self._dram_lru, expert_id)
            if expert_id in self._optims:
                del self._optims[expert_id]
            self.experts[expert_id].to("meta")
            return

        if target.type == "cuda" and init_dev.type != "cuda":
            # Non-blocking move to GPU (counting bytes/timing as a stall).
            self._ensure_on_gpu(expert_id)
        elif target.type == "cpu" and init_dev.type != "cpu":
            self._ensure_on_cpu(expert_id)

        # Touch appropriate LRU and enforce budgets.
        if self._is_on_gpu(expert_id):
            self._touch_lru(self._hbm_lru, expert_id)
            self._evict_hbm_if_needed(protect=set([expert_id]))
        else:
            self._touch_lru(self._dram_lru, expert_id)
            self._evict_dram_if_needed()


    def save_experts_to_disk(self, expert_ids: Iterable[int]) -> None:
        """Persist selected experts to disk (weights + optimizer state if enabled)."""
        if not self.enable_disk:
            return
        ids = sorted(set(int(x) for x in expert_ids))
        for eid in ids:
            self._ensure_on_cpu(eid)
            self._save_to_disk(eid)

    def reset_after_night(
        self,
        *,
        retain_cpu_ids: Sequence[int],
        writeback_ids: Optional[Iterable[int]] = None,
        clear_hbm: bool = True,
    ) -> None:
        """End-of-night behavior.

        - Optionally write back `writeback_ids` to disk
        - Keep `retain_cpu_ids` resident on CPU (DRAM)
        - Drop everything else currently resident (HBM/DRAM) to meta to free memory
        - Optionally clear the HBM tier

        Requires enable_disk=True; otherwise dropping to meta would lose weights.
        """
        if not self.enable_disk:
            return

        # Ensure we're not in a pinned region.
        self._pin_active = False
        self._pinned_hbm.clear()

        # Unique, stable order.
        retain_list: List[int] = []
        seen: Set[int] = set()
        for x in retain_cpu_ids:
            eid = int(x)
            if eid not in seen:
                seen.add(eid)
                retain_list.append(eid)
        retain_set = set(retain_list)

        # Ensure retained experts are materialized on CPU (and counted as DRAM-resident).
        if retain_list:
            self.ensure_on_cpu(retain_list)

        # Write back dirty experts (best-effort).
        if writeback_ids is not None:
            self.save_experts_to_disk(writeback_ids)

        # Collect resident experts (HBM + DRAM).
        resident = set(self._hbm_lru + self._dram_lru)

        # Drop non-retained experts to meta to free memory.
        for eid in sorted(resident):
            if eid in retain_set:
                continue

            # If on GPU, move to CPU first (then to meta), so we don't keep CUDA allocations around.
            if self._is_on_gpu(eid):
                e = self.experts[eid]
                self._remove_lru(self._hbm_lru, eid)
                d2h = sum(int(p.numel() * p.element_size()) for p in e.parameters())
                t0 = time.perf_counter()
                e = e.to("cpu")
                dt = time.perf_counter() - t0
                self.stats.bytes_d2h += int(d2h)
                self.stats.stall_time_s += float(dt)

                if self._can_pin:
                    for p in e.parameters():
                        if p.device.type == "cpu":
                            p.data = p.data.pin_memory()

                if self.create_expert_optimizer and eid in self._optims:
                    _move_optimizer_state_(self._optims[eid], torch.device("cpu"))

            # Remove from DRAM LRU if present.
            self._remove_lru(self._dram_lru, eid)

            # Drop optimizer state for this expert.
            if eid in self._optims:
                del self._optims[eid]

            # Move to meta to free memory.
            self.experts[eid].to("meta")

        # Clear HBM tier if requested (retained experts should already be on CPU).
        if clear_hbm:
            self._hbm_lru.clear()

        # Rebuild DRAM LRU to contain only retained experts (MRU order as retain_list).
        self._dram_lru = [eid for eid in retain_list if eid in retain_set]

        # Enforce DRAM capacity if caller provided an oversized list.
        self._evict_dram_if_needed()

    def flush_resident_to_disk(self) -> None:
        """Best-effort: persist all currently materialized experts to disk.

        Useful in Colab so you don't lose expert weights/optim state if the runtime dies.

        Safety:
        - Never attempt to save an expert that is currently on the meta device (no storage).
        - If an expert ended up meta but is still listed as resident, prune it from the LRU lists.
        """
        if not self.enable_disk:
            return

        ids = set(self._hbm_lru + self._dram_lru)

        pruned_meta = 0
        for eid in sorted(ids):
            if self._is_on_meta(eid):
                # Stale bookkeeping: meta experts should not be counted as "resident".
                self._remove_lru(self._hbm_lru, eid)
                self._remove_lru(self._dram_lru, eid)
                pruned_meta += 1
                continue
            self._save_to_disk(eid)

        if pruned_meta > 0:
            print(f"[ExpertStore] NOTE: pruned {pruned_meta} stale meta experts from resident LRU during flush.")

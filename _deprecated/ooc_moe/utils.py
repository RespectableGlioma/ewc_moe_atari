
from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Generic, Hashable, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

@contextlib.contextmanager
def timed(stats: "Stats", key: str) -> Iterator[None]:
    """
    Context manager to accumulate wall-clock timing into stats.timers[key].
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        stats.add_time(key, dt)

@dataclass
class Stats:
    """
    Lightweight instrumentation container.

    This object may be updated from background I/O threads (prefetch / writeback).
    We keep internal locking so counters/timers remain consistent.
    """
    timers: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    bytes_moved: Dict[str, int] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def add_bytes(self, key: str, nbytes: int) -> None:
        with self._lock:
            self.bytes_moved[key] = self.bytes_moved.get(key, 0) + int(nbytes)

    def add_time(self, key: str, seconds: float) -> None:
        with self._lock:
            self.timers[key] = self.timers.get(key, 0.0) + float(seconds)

    def summary(self) -> str:
        lines = ["== Stats =="]
        if self.counters:
            lines.append("-- counters --")
            for k, v in sorted(self.counters.items()):
                lines.append(f"{k}: {v}")
        if self.bytes_moved:
            lines.append("-- bytes --")
            for k, v in sorted(self.bytes_moved.items()):
                lines.append(f"{k}: {v/1e9:.3f} GB")
        if self.timers:
            lines.append("-- timers --")
            for k, v in sorted(self.timers.items()):
                lines.append(f"{k}: {v:.3f} s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Dict[str, float] | Dict[str, int]]:
        """Serialize stats for JSON logging."""
        # Copy under lock to avoid partial reads.
        with self._lock:
            return {
                "counters": dict(self.counters),
                "timers": dict(self.timers),
                "bytes_moved": dict(self.bytes_moved),
            }


class LRUCache(Generic[K, V]):
    """
    A small LRU cache with explicit eviction.
    """
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._od: "OrderedDict[K, V]" = OrderedDict()

    def __contains__(self, key: K) -> bool:
        return key in self._od

    def get(self, key: K) -> Optional[V]:
        if key not in self._od:
            return None
        v = self._od.pop(key)
        self._od[key] = v
        return v

    def put(self, key: K, value: V) -> Tuple[Optional[Tuple[K, V]], bool]:
        """
        Insert key/value. Returns (evicted_item, replaced_existing).
        evicted_item is (old_key, old_value) if eviction occurred.
        """
        replaced = False
        if key in self._od:
            self._od.pop(key)
            replaced = True
        self._od[key] = value
        evicted = None
        if self.capacity == 0:
            # evict immediately
            evicted = self._od.popitem(last=False)
        elif len(self._od) > self.capacity:
            evicted = self._od.popitem(last=False)
        return evicted, replaced

    def pop(self, key: K) -> Optional[V]:
        return self._od.pop(key, None)

    def items(self):
        return self._od.items()

    def keys(self):
        return self._od.keys()

    def __len__(self) -> int:
        return len(self._od)

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self.capacity}, size={len(self._od)})"

"""
TieredStore: Manages expert storage on disk (NVMe).

Handles serialization, compression, and async loading of experts.
In production, this would interface with NVMe directly for optimal throughput.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logger = logging.getLogger(__name__)


class TieredStore:
    """
    Tiered storage manager for expert library.

    Hierarchy:
    1. GPU HBM (managed by ExpertManager)
    2. CPU RAM (for prefetching)
    3. NVMe/SSD (persistent storage)

    This class manages levels 2-3.
    """

    def __init__(
        self,
        base_path: str = "./expert_library",
        cpu_cache_size: int = 4,  # Number of experts to keep in CPU RAM
        num_workers: int = 2,  # Async loading threads
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.experts_path = self.base_path / "experts"
        self.experts_path.mkdir(exist_ok=True)

        self.registry_path = self.base_path / "registry.json"

        self.cpu_cache_size = cpu_cache_size
        self.num_workers = num_workers

        # CPU RAM cache (LRU-style)
        self.cpu_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.cache_lock = threading.Lock()

        # Async loading
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.pending_loads: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            'disk_reads': 0,
            'disk_writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'bytes_read': 0,
            'bytes_written': 0,
        }

    def save_expert(
        self,
        expert_id: str,
        data: Dict[str, Any],
        async_save: bool = False,
    ):
        """
        Save expert data to disk.

        Args:
            expert_id: unique identifier for the expert
            data: state dict with metadata
            async_save: if True, save in background thread
        """
        if async_save:
            self.executor.submit(self._save_expert_sync, expert_id, data)
        else:
            self._save_expert_sync(expert_id, data)

    def _save_expert_sync(self, expert_id: str, data: Dict[str, Any]):
        """Synchronous expert save."""
        filepath = self.experts_path / f"{expert_id}.pt"

        # Move tensors to CPU before saving
        save_data = self._prepare_for_save(data)

        torch.save(save_data, filepath)

        self.stats['disk_writes'] += 1
        self.stats['bytes_written'] += filepath.stat().st_size

        # Update CPU cache
        with self.cache_lock:
            self._add_to_cache(expert_id, save_data)

        logger.debug(f"Saved expert {expert_id} ({filepath.stat().st_size / 1024:.1f} KB)")

    def _prepare_for_save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for saving (move to CPU)."""
        save_data = {}
        for key, value in data.items():
            if key == 'state_dict':
                save_data[key] = {
                    k: v.cpu() for k, v in value.items()
                }
            else:
                save_data[key] = value
        return save_data

    def load_expert(
        self,
        expert_id: str,
        async_load: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Load expert data from storage.

        Checks CPU cache first, then disk.

        Args:
            expert_id: unique identifier for the expert
            async_load: if True, return future instead of data

        Returns:
            Expert data dict, or None if not found
        """
        # Check CPU cache first
        with self.cache_lock:
            if expert_id in self.cpu_cache:
                self.cache_access_times[expert_id] = time.time()
                self.stats['cache_hits'] += 1
                logger.debug(f"CPU cache hit for {expert_id}")
                return self.cpu_cache[expert_id]

        self.stats['cache_misses'] += 1

        if async_load:
            if expert_id not in self.pending_loads:
                future = self.executor.submit(self._load_expert_sync, expert_id)
                self.pending_loads[expert_id] = future
            return self.pending_loads.get(expert_id)

        return self._load_expert_sync(expert_id)

    def _load_expert_sync(self, expert_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous expert load."""
        filepath = self.experts_path / f"{expert_id}.pt"

        if not filepath.exists():
            logger.warning(f"Expert file not found: {filepath}")
            return None

        data = torch.load(filepath, map_location='cpu', weights_only=False)

        self.stats['disk_reads'] += 1
        self.stats['bytes_read'] += filepath.stat().st_size

        # Add to CPU cache
        with self.cache_lock:
            self._add_to_cache(expert_id, data)

        # Remove from pending if present
        self.pending_loads.pop(expert_id, None)

        logger.debug(f"Loaded expert {expert_id} from disk")
        return data

    def _add_to_cache(self, expert_id: str, data: Dict[str, Any]):
        """Add expert to CPU cache, evicting if necessary."""
        # Evict oldest if cache full
        while len(self.cpu_cache) >= self.cpu_cache_size:
            oldest_id = min(self.cache_access_times, key=self.cache_access_times.get)
            del self.cpu_cache[oldest_id]
            del self.cache_access_times[oldest_id]
            logger.debug(f"Evicted {oldest_id} from CPU cache")

        self.cpu_cache[expert_id] = data
        self.cache_access_times[expert_id] = time.time()

    def prefetch_expert(self, expert_id: str):
        """Start async loading of expert into CPU cache."""
        with self.cache_lock:
            if expert_id in self.cpu_cache:
                return  # Already cached

        if expert_id not in self.pending_loads:
            future = self.executor.submit(self._load_expert_sync, expert_id)
            self.pending_loads[expert_id] = future
            logger.debug(f"Started prefetching {expert_id}")

    def wait_for_prefetch(self, expert_id: str, timeout: float = 1.0) -> bool:
        """Wait for prefetch to complete."""
        future = self.pending_loads.get(expert_id)
        if future is None:
            return expert_id in self.cpu_cache

        try:
            future.result(timeout=timeout)
            return True
        except Exception:
            return False

    def save_registry(self, registry: Dict[str, Any]):
        """Save expert registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.debug("Saved registry")

    def load_registry(self) -> Optional[Dict[str, Any]]:
        """Load expert registry from disk."""
        if not self.registry_path.exists():
            return None

        with open(self.registry_path, 'r') as f:
            return json.load(f)

    def list_experts(self) -> list:
        """List all stored expert IDs."""
        return [f.stem for f in self.experts_path.glob("*.pt")]

    def delete_expert(self, expert_id: str):
        """Delete expert from storage."""
        filepath = self.experts_path / f"{expert_id}.pt"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted expert {expert_id}")

        with self.cache_lock:
            self.cpu_cache.pop(expert_id, None)
            self.cache_access_times.pop(expert_id, None)

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_size = sum(f.stat().st_size for f in self.experts_path.glob("*.pt"))
        num_experts = len(list(self.experts_path.glob("*.pt")))

        return {
            **self.stats,
            'total_experts': num_experts,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_kb': (total_size / num_experts / 1024) if num_experts > 0 else 0,
            'cpu_cache_size': len(self.cpu_cache),
        }

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("TieredStore cleanup complete")

"""
Tiered Expert Store: Manages experts across HBM/DRAM/NVMe with prefetching.

This is the core abstraction for out-of-core MOE training. It provides:
1. LRU caching across memory tiers
2. Asynchronous prefetching
3. Access pattern tracking for learning
4. Simulated latency for prototyping (DRAM acts as "slow storage")

The key insight: if we can predict which experts will be needed,
we can hide the latency of slower storage tiers through prefetching.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock, Thread
from queue import Queue, Empty
import time
import numpy as np
from enum import Enum


class StorageTier(Enum):
    HBM = "hbm"      # GPU memory - fastest
    DRAM = "dram"    # CPU memory - medium
    NVME = "nvme"    # Disk - slowest


@dataclass
class TierConfig:
    """Configuration for a storage tier."""
    capacity: int  # Number of experts this tier can hold
    read_latency_ms: float  # Simulated read latency in milliseconds
    write_latency_ms: float  # Simulated write latency in milliseconds
    bandwidth_gbps: float  # Bandwidth in GB/s (for size-based latency)


@dataclass
class AccessRecord:
    """Record of an expert access for pattern learning."""
    expert_id: int
    timestamp: float
    tier_accessed: StorageTier
    was_cache_hit: bool
    context_hash: Optional[int] = None  # Hash of input context for correlation


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    prefetch_hits: int = 0  # Hits that were prefetched
    evictions: int = 0
    tier_accesses: Dict[StorageTier, int] = field(default_factory=dict)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def prefetch_effectiveness(self) -> float:
        return self.prefetch_hits / self.hits if self.hits > 0 else 0.0


class LRUCache:
    """
    Thread-safe LRU cache for expert tensors.
    
    Uses OrderedDict for O(1) access and LRU ordering.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, Tensor] = OrderedDict()
        self.lock = Lock()
        self.prefetched: Set[int] = set()  # Track which items were prefetched
        
    def get(self, key: int) -> Optional[Tuple[Tensor, bool]]:
        """
        Get item from cache.
        Returns (tensor, was_prefetched) or None if not found.
        """
        with self.lock:
            if key not in self.cache:
                return None
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            was_prefetched = key in self.prefetched
            self.prefetched.discard(key)
            return self.cache[key], was_prefetched
    
    def put(self, key: int, value: Tensor, is_prefetch: bool = False) -> Optional[int]:
        """
        Put item in cache.
        Returns evicted key if cache was full, None otherwise.
        """
        with self.lock:
            evicted = None
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.capacity:
                    # Evict least recently used
                    evicted, _ = self.cache.popitem(last=False)
                    self.prefetched.discard(evicted)
                self.cache[key] = value
            
            if is_prefetch:
                self.prefetched.add(key)
            
            return evicted
    
    def contains(self, key: int) -> bool:
        with self.lock:
            return key in self.cache
    
    def keys(self) -> List[int]:
        with self.lock:
            return list(self.cache.keys())
    
    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)
    
    def __contains__(self, key: int) -> bool:
        return self.contains(key)


class TieredExpertStore:
    """
    Manages expert parameters across a tiered memory hierarchy.
    
    Architecture:
    - HBM (GPU): Fast cache, limited capacity
    - DRAM (CPU): Medium cache, larger capacity  
    - NVMe (Disk): Cold storage, largest capacity
    
    For prototyping, we simulate the hierarchy using:
    - HBM: CUDA tensors
    - DRAM: CPU tensors with artificial latency
    - NVMe: CPU tensors with higher artificial latency
    
    Key features:
    1. Asynchronous prefetching based on predictions
    2. Access pattern tracking for learning
    3. Gradient accumulation across tiers
    """
    
    def __init__(
        self,
        num_experts: int,
        expert_dim: int,
        hidden_dim: int,
        hbm_capacity: int = 32,
        dram_capacity: int = 128,
        device: str = "cuda",
        simulate_latency: bool = True,
        hbm_latency_ms: float = 0.0,
        dram_latency_ms: float = 1.0,
        nvme_latency_ms: float = 10.0,
    ):
        """
        Initialize tiered expert store.
        
        Args:
            num_experts: Total number of experts
            expert_dim: Input/output dimension of each expert
            hidden_dim: Hidden dimension within each expert (FFN)
            hbm_capacity: Max experts in HBM
            dram_capacity: Max experts in DRAM
            device: Primary compute device
            simulate_latency: Whether to simulate storage latency
            hbm_latency_ms: Simulated HBM access latency
            dram_latency_ms: Simulated DRAM access latency  
            nvme_latency_ms: Simulated NVMe access latency
        """
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.simulate_latency = simulate_latency
        
        # Tier configurations
        self.tier_configs = {
            StorageTier.HBM: TierConfig(hbm_capacity, hbm_latency_ms, hbm_latency_ms, 3000),
            StorageTier.DRAM: TierConfig(dram_capacity, dram_latency_ms, dram_latency_ms, 200),
            StorageTier.NVME: TierConfig(num_experts, nvme_latency_ms, nvme_latency_ms, 7),
        }
        
        # Caches for each tier
        self.hbm_cache = LRUCache(hbm_capacity)
        self.dram_cache = LRUCache(dram_capacity)
        
        # Cold storage (all experts live here as ground truth)
        # In real implementation, this would be memory-mapped from NVMe
        self.nvme_store: Dict[int, Dict[str, Tensor]] = {}
        
        # Initialize all experts in cold storage
        self._initialize_experts()
        
        # Prefetch queue and worker
        self.prefetch_queue: Queue = Queue()
        self.prefetch_worker: Optional[Thread] = None
        self.stop_prefetch = False
        
        # Statistics and access history
        self.stats = CacheStats()
        self.stats.tier_accesses = {tier: 0 for tier in StorageTier}
        self.access_history: List[AccessRecord] = []
        self.max_history_len = 100000
        
        # Gradient accumulators for cold experts
        self.gradient_accumulators: Dict[int, Dict[str, Tensor]] = {}
        self.gradient_counts: Dict[int, int] = {}
        
        # Start prefetch worker
        self._start_prefetch_worker()
    
    def _initialize_experts(self):
        """Initialize all experts in cold storage with random weights."""
        for i in range(self.num_experts):
            # Each expert is a 2-layer FFN: dim -> hidden -> dim
            self.nvme_store[i] = {
                'w1': torch.randn(self.hidden_dim, self.expert_dim) * 0.02,
                'w2': torch.randn(self.expert_dim, self.hidden_dim) * 0.02,
                'b1': torch.zeros(self.hidden_dim),
                'b2': torch.zeros(self.expert_dim),
            }
    
    def _start_prefetch_worker(self):
        """Start background thread for prefetching."""
        self.stop_prefetch = False
        self.prefetch_worker = Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_worker.start()
    
    def _prefetch_loop(self):
        """Background loop that processes prefetch requests."""
        while not self.stop_prefetch:
            try:
                expert_id = self.prefetch_queue.get(timeout=0.1)
                self._promote_expert(expert_id, is_prefetch=True)
            except Empty:
                continue
    
    def _simulate_latency(self, tier: StorageTier):
        """Simulate storage access latency."""
        if self.simulate_latency:
            latency = self.tier_configs[tier].read_latency_ms / 1000.0
            if latency > 0:
                time.sleep(latency)
    
    def _promote_expert(self, expert_id: int, is_prefetch: bool = False) -> Dict[str, Tensor]:
        """
        Promote an expert from cold storage to HBM.
        
        This implements the caching hierarchy:
        1. Check HBM (fastest)
        2. Check DRAM (medium)
        3. Load from NVMe (slowest)
        
        Returns expert parameters on the target device.
        """
        # Check HBM
        result = self.hbm_cache.get(expert_id)
        if result is not None:
            params, was_prefetched = result
            self.stats.hits += 1
            if was_prefetched:
                self.stats.prefetch_hits += 1
            self.stats.tier_accesses[StorageTier.HBM] += 1
            return params
        
        # Check DRAM
        result = self.dram_cache.get(expert_id)
        if result is not None:
            params_cpu, was_prefetched = result
            self._simulate_latency(StorageTier.DRAM)
            
            # Promote to HBM
            params_gpu = {k: v.to(self.device) for k, v in params_cpu.items()}
            evicted = self.hbm_cache.put(expert_id, params_gpu, is_prefetch=is_prefetch)
            
            # Demote evicted expert to DRAM
            if evicted is not None:
                evicted_result = self.hbm_cache.get(evicted)
                if evicted_result:
                    evicted_params, _ = evicted_result
                    evicted_cpu = {k: v.cpu() for k, v in evicted_params.items()}
                    dram_evicted = self.dram_cache.put(evicted, evicted_cpu)
                    if dram_evicted is not None:
                        self.stats.evictions += 1
            
            self.stats.hits += 1
            if was_prefetched:
                self.stats.prefetch_hits += 1
            self.stats.tier_accesses[StorageTier.DRAM] += 1
            return params_gpu
        
        # Load from NVMe (cold storage)
        self._simulate_latency(StorageTier.NVME)
        params_cold = self.nvme_store[expert_id]
        params_gpu = {k: v.to(self.device) for k, v in params_cold.items()}
        
        # Cache in HBM
        evicted = self.hbm_cache.put(expert_id, params_gpu, is_prefetch=is_prefetch)
        
        # Handle eviction cascade
        if evicted is not None:
            evicted_params = self._get_from_hbm_direct(evicted)
            if evicted_params:
                evicted_cpu = {k: v.cpu() for k, v in evicted_params.items()}
                dram_evicted = self.dram_cache.put(evicted, evicted_cpu)
                if dram_evicted is not None:
                    # Write back to NVMe if dirty
                    self._writeback_to_nvme(dram_evicted)
                    self.stats.evictions += 1
        
        self.stats.misses += 1
        self.stats.tier_accesses[StorageTier.NVME] += 1
        return params_gpu
    
    def _get_from_hbm_direct(self, expert_id: int) -> Optional[Dict[str, Tensor]]:
        """Get expert from HBM without updating LRU order."""
        with self.hbm_cache.lock:
            if expert_id in self.hbm_cache.cache:
                return self.hbm_cache.cache[expert_id]
        return None
    
    def _writeback_to_nvme(self, expert_id: int):
        """Write expert parameters back to cold storage."""
        # In a real implementation, this would be async I/O
        result = self.dram_cache.get(expert_id)
        if result is not None:
            params, _ = result
            self.nvme_store[expert_id] = {k: v.clone() for k, v in params.items()}
    
    def get_experts(
        self, 
        expert_ids: List[int],
        context_hash: Optional[int] = None
    ) -> Dict[int, Dict[str, Tensor]]:
        """
        Fetch multiple experts, potentially waiting for prefetch.
        
        Args:
            expert_ids: List of expert IDs to fetch
            context_hash: Optional hash of input context for access tracking
            
        Returns:
            Dictionary mapping expert_id -> {w1, w2, b1, b2}
        """
        results = {}
        timestamp = time.time()
        
        for eid in expert_ids:
            params = self._promote_expert(eid)
            results[eid] = params
            
            # Record access
            tier = self._get_expert_tier(eid)
            was_hit = tier == StorageTier.HBM
            self.access_history.append(AccessRecord(
                expert_id=eid,
                timestamp=timestamp,
                tier_accessed=tier,
                was_cache_hit=was_hit,
                context_hash=context_hash
            ))
        
        # Trim history if needed
        if len(self.access_history) > self.max_history_len:
            self.access_history = self.access_history[-self.max_history_len:]
        
        return results
    
    def _get_expert_tier(self, expert_id: int) -> StorageTier:
        """Determine which tier an expert is currently in."""
        if expert_id in self.hbm_cache:
            return StorageTier.HBM
        elif expert_id in self.dram_cache:
            return StorageTier.DRAM
        return StorageTier.NVME
    
    def prefetch(self, expert_ids: List[int]):
        """
        Non-blocking prefetch of experts into HBM.
        
        Called by the environment detector to warm the cache
        before experts are needed.
        """
        for eid in expert_ids:
            if eid not in self.hbm_cache:
                self.prefetch_queue.put(eid)
    
    def accumulate_gradients(
        self,
        expert_id: int,
        gradients: Dict[str, Tensor]
    ):
        """
        Accumulate gradients for an expert.
        
        For cold experts, we accumulate gradients and apply them
        in batched updates to amortize I/O cost.
        """
        if expert_id not in self.gradient_accumulators:
            self.gradient_accumulators[expert_id] = {
                k: torch.zeros_like(v) for k, v in gradients.items()
            }
            self.gradient_counts[expert_id] = 0
        
        for k, grad in gradients.items():
            self.gradient_accumulators[expert_id][k] += grad.cpu()
        self.gradient_counts[expert_id] += 1
    
    def apply_accumulated_gradients(self, lr: float, min_accumulations: int = 10):
        """
        Apply accumulated gradients to cold storage.
        
        Only applies if enough gradients have accumulated to
        amortize the I/O cost.
        """
        experts_updated = 0
        for expert_id, grads in list(self.gradient_accumulators.items()):
            count = self.gradient_counts.get(expert_id, 0)
            if count >= min_accumulations:
                # Average the gradients
                for k in grads:
                    grads[k] /= count
                    # Apply to cold storage
                    self.nvme_store[expert_id][k] -= lr * grads[k]
                
                # Clear accumulators
                del self.gradient_accumulators[expert_id]
                del self.gradient_counts[expert_id]
                
                # Invalidate cached copies
                # (simplified - real impl would be more careful)
                experts_updated += 1
        
        return experts_updated
    
    def get_access_patterns(self, last_n: int = 1000) -> np.ndarray:
        """
        Get recent access patterns as a numpy array for analysis.
        
        Returns array of shape (n, 2): [expert_id, timestamp]
        """
        recent = self.access_history[-last_n:]
        if not recent:
            return np.array([])
        return np.array([[r.expert_id, r.timestamp] for r in recent])
    
    def get_expert_cooccurrence(self, window_size: int = 100) -> Dict[Tuple[int, int], int]:
        """
        Compute expert co-occurrence within temporal windows.
        
        Useful for learning which experts are accessed together.
        """
        cooccurrence: Dict[Tuple[int, int], int] = {}
        recent = self.access_history[-10000:]
        
        for i in range(0, len(recent) - window_size, window_size):
            window = recent[i:i + window_size]
            experts_in_window = set(r.expert_id for r in window)
            for e1 in experts_in_window:
                for e2 in experts_in_window:
                    if e1 < e2:
                        key = (e1, e2)
                        cooccurrence[key] = cooccurrence.get(key, 0) + 1
        
        return cooccurrence
    
    def get_stats_summary(self) -> Dict:
        """Get summary statistics for monitoring."""
        return {
            'hit_rate': self.stats.hit_rate,
            'prefetch_effectiveness': self.stats.prefetch_effectiveness,
            'total_hits': self.stats.hits,
            'total_misses': self.stats.misses,
            'prefetch_hits': self.stats.prefetch_hits,
            'evictions': self.stats.evictions,
            'hbm_occupancy': len(self.hbm_cache) / self.tier_configs[StorageTier.HBM].capacity,
            'dram_occupancy': len(self.dram_cache) / self.tier_configs[StorageTier.DRAM].capacity,
            'tier_accesses': {t.value: c for t, c in self.stats.tier_accesses.items()},
        }
    
    def shutdown(self):
        """Clean shutdown of prefetch worker."""
        self.stop_prefetch = True
        if self.prefetch_worker:
            self.prefetch_worker.join(timeout=2.0)


class ExpertParameterServer:
    """
    Higher-level interface for training with tiered experts.
    
    Wraps TieredExpertStore with training-specific functionality:
    - Parameter synchronization
    - Optimizer state management
    - Checkpoint save/load
    """
    
    def __init__(self, store: TieredExpertStore):
        self.store = store
        self.optimizer_states: Dict[int, Dict] = {}
    
    def get_expert_module(self, expert_id: int) -> nn.Module:
        """
        Get a trainable nn.Module wrapper for an expert.
        
        The module's parameters are views into cached storage.
        """
        params = self.store._promote_expert(expert_id)
        
        class ExpertModule(nn.Module):
            def __init__(self, w1, w2, b1, b2):
                super().__init__()
                self.w1 = nn.Parameter(w1)
                self.w2 = nn.Parameter(w2)
                self.b1 = nn.Parameter(b1)
                self.b2 = nn.Parameter(b2)
            
            def forward(self, x):
                h = torch.relu(x @ self.w1.T + self.b1)
                return h @ self.w2.T + self.b2
        
        return ExpertModule(params['w1'], params['w2'], params['b1'], params['b2'])
    
    def save_checkpoint(self, path: str):
        """Save all expert parameters to disk."""
        torch.save({
            'experts': self.store.nvme_store,
            'stats': self.store.stats,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load expert parameters from disk."""
        checkpoint = torch.load(path)
        self.store.nvme_store = checkpoint['experts']
        self.store.stats = checkpoint.get('stats', CacheStats())

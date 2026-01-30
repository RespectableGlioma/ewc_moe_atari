
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

from .tiered_store import TieredExpertStore


@torch.no_grad()
def route_fixed_by_env(env_ids: torch.Tensor, n_experts: int, experts_per_env: int = 1) -> torch.Tensor:
    """
    Deterministic routing:
      - If experts_per_env == 1: expert = env_id % n_experts
      - Else: pick among a small environment-specific set (cyclic).
    """
    if env_ids.dtype != torch.long:
        env_ids = env_ids.long()
    if experts_per_env <= 1:
        return env_ids % n_experts
    # A simple hash to pick within an env-specific expert block
    base = (env_ids * experts_per_env) % n_experts
    # Here we just return base; training loop can add jitter if desired.
    return base


def moe_forward_top1(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    store: TieredExpertStore,
    *,
    sort_by_expert: bool = True,
) -> torch.Tensor:
    """
    Top-1 MoE forward using tiered expert store.

    x: [B, d_model]
    expert_ids: [B] int64
    returns: [B, d_model]
    """
    if expert_ids.dtype != torch.long:
        expert_ids = expert_ids.long()

    B, d = x.shape
    y = torch.empty((B, d), device=x.device, dtype=x.dtype)

    if sort_by_expert:
        # Sort tokens by expert for better cache locality.
        sorted_ids, perm = torch.sort(expert_ids)
        x_sorted = x.index_select(0, perm)
        y_sorted = torch.empty_like(x_sorted)

        # Find segment boundaries for each expert.
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
        offsets = torch.cumsum(counts, dim=0)
        start = 0
        for eid, end in zip(unique_ids.tolist(), offsets.tolist()):
            slot, _ = store.ensure_on_gpu(int(eid))
            y_sorted[start:end] = slot.forward(x_sorted[start:end])
            start = end

        # Unsort
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(B, device=perm.device)
        y = y_sorted.index_select(0, inv)
        return y

    # Unsorted path (simpler but potentially worse locality)
    unique = torch.unique(expert_ids).tolist()
    for eid in unique:
        idx = (expert_ids == eid).nonzero(as_tuple=False).squeeze(-1)
        slot, _ = store.ensure_on_gpu(int(eid))
        y.index_copy_(0, idx, slot.forward(x.index_select(0, idx)))
    return y

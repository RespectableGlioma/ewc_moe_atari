from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Optional

import torch
import torch.nn as nn


FilterFn = Optional[Callable[[str], bool]]
LossFn = Callable[[nn.Module, object], torch.Tensor]


@dataclass
class TaskSnapshot:
    """EWC snapshot for a single task/domain."""
    name: str
    params_star: Dict[str, torch.Tensor]   # CPU tensors
    fisher_diag: Dict[str, torch.Tensor]   # CPU tensors


def estimate_fisher_diag(
    model: nn.Module,
    *,
    batches: Iterable[object],
    loss_fn: LossFn,
    filter_fn: FilterFn = None,
    max_batches: int = 25,
    begin_step_fn: Optional[Callable[[], None]] = None,
    end_step_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Fisher information (approx) via squared gradients:
        F ≈ E[ (∂L/∂θ)^2 ]

    Key design choices for out-of-core / paging:
    - We ALWAYS accumulate Fisher buffers on CPU to avoid device-mismatch issues
      when experts page CPU<->GPU between batches.
    - We DO NOT toggle requires_grad flags for parameters (that can accidentally
      freeze the whole model if name filters mismatch and produce a loss with
      no grad_fn). Instead we compute grads normally and only *accumulate*
      for parameters that pass filter_fn.
    - We wrap each batch in torch.enable_grad() so this works even if called from
      a no_grad() context higher up.
    - If begin_step_fn/end_step_fn are provided (ExpertStore pinning), we call
      begin_step_fn() before forward and end_step_fn() only AFTER we've copied
      gradients into CPU buffers.
    """
    if max_batches <= 0:
        return {}

    was_training = model.training
    # Keep model mode as-is; Fisher is meant to reflect training dynamics.
    # We only ensure grad is enabled.
    fisher: Dict[str, torch.Tensor] = {}
    n = 0

    try:
        with torch.enable_grad():
            for batch in batches:
                if n >= max_batches:
                    break

                if begin_step_fn is not None:
                    begin_step_fn()

                try:
                    # Clear grads
                    model.zero_grad(set_to_none=True)

                    loss = loss_fn(model, batch)
                    if not torch.is_tensor(loss):
                        # Defensive: allow loss_fn to return python floats (shouldn't happen)
                        loss = torch.as_tensor(loss, device=next(model.parameters()).device)

                    # Ensure scalar
                    if loss.ndim != 0:
                        loss = loss.mean()

                    # If loss doesn't require grad, backprop will fail. This can happen
                    # if someone wrapped Fisher in inference_mode/no_grad, or if all
                    # parameters were frozen externally. We try to continue gracefully.
                    if not loss.requires_grad:
                        # Still count the batch so we don't infinite loop on weird iterators.
                        n += 1
                        continue

                    loss.backward()

                    # Accumulate squared grads on CPU.
                    for name, p in model.named_parameters():
                        if filter_fn is not None and not filter_fn(name):
                            continue
                        if p.grad is None:
                            continue
                        g2_cpu = p.grad.detach().float().cpu().pow(2)
                        if name not in fisher:
                            fisher[name] = torch.zeros_like(g2_cpu, device="cpu")
                        fisher[name] += g2_cpu

                    n += 1

                finally:
                    # IMPORTANT: only unpin/evict after we've read grads.
                    if end_step_fn is not None:
                        end_step_fn()

    finally:
        model.train(was_training)

    # Normalize
    denom = max(1, n)
    for k in list(fisher.keys()):
        fisher[k] = fisher[k] / float(denom)

    return fisher


def make_snapshot(
    model: nn.Module,
    *,
    name: str,
    fisher_diag: Dict[str, torch.Tensor],
    filter_fn: FilterFn = None,
) -> TaskSnapshot:
    """
    Create a CPU snapshot of parameters and Fisher diag for EWC penalty.
    """
    params_star: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if filter_fn is not None and not filter_fn(n):
                continue
            # store CPU copy
            params_star[n] = p.detach().cpu().clone()

    # Ensure fisher is CPU + detached
    fisher_cpu: Dict[str, torch.Tensor] = {}
    for n, f in fisher_diag.items():
        if filter_fn is not None and not filter_fn(n):
            continue
        fisher_cpu[n] = f.detach().cpu().clone()

    return TaskSnapshot(name=name, params_star=params_star, fisher_diag=fisher_cpu)


class EWC:
    """
    Elastic Weight Consolidation penalty:
        penalty = (λ/2) * Σ_tasks Σ_params F_i (θ_i - θ*_i)^2
    where F_i is diagonal Fisher estimate and θ* is the snapshot parameter value.
    """
    def __init__(self, *, lambda_: float = 0.4):
        self.lambda_ = float(lambda_)
        self.tasks: list[TaskSnapshot] = []

    def add_task_snapshot(self, snap: TaskSnapshot) -> None:
        self.tasks.append(snap)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self.tasks or self.lambda_ <= 0.0:
            # Make sure penalty is on the right device
            dev = next(model.parameters()).device
            return torch.zeros((), device=dev)

        dev = next(model.parameters()).device
        total = torch.zeros((), device=dev)

        # We iterate model params once per task. This is fine at small scale and
        # avoids needing to keep GPU copies of all snapshots.
        for task in self.tasks:
            for n, p in model.named_parameters():
                if n not in task.fisher_diag:
                    continue
                f = task.fisher_diag[n].to(dev, non_blocking=True)
                p_star = task.params_star[n].to(dev, non_blocking=True)
                total = total + (f * (p - p_star).pow(2)).sum()

        return 0.5 * self.lambda_ * total

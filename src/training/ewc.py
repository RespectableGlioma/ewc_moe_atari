
from __future__ import annotations

# EWC utilities for continual learning in the day/night MoE Atari scaffold.
#
# This file is intentionally defensive for out-of-core expert paging:
# - Experts can be on GPU/CPU/meta depending on the cache tier.
# - We avoid toggling requires_grad flags (a common source of "loss has no grad_fn").
# - Fisher accumulation is always done on CPU.
# - If a parameter lives on the 'meta' device, we skip it (can't compute penalty / snapshot).
#
# Sentinel for debugging:
EWC_VERSION = "EWC_V12_GRADFIX"

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Optional

import torch
import torch.nn as nn


FilterFn = Optional[Callable[[str], bool]]


@dataclass
class TaskSnapshot:
    name: str
    params_star: Dict[str, torch.Tensor]   # CPU tensors
    fisher_diag: Dict[str, torch.Tensor]   # CPU tensors


def estimate_fisher_diag(
    model: nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    *,
    loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    filter_fn: FilterFn = None,
    max_batches: int = 25,
    begin_step_fn: Optional[Callable[[], None]] = None,
    end_step_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate a diagonal Fisher approximation via squared gradients.

    Robustness goals:
    - Never crash if a batch produces a loss without grad_fn (skip it).
    - Never depend on parameter device stability (accumulate fisher on CPU).
    - Never mutate requires_grad flags (can accidentally freeze everything).
    """
    fisher: Dict[str, torch.Tensor] = {}
    n_ok = 0
    n_skipped_no_grad = 0

    # Ensure model is in train mode for gradient computation; this is typical for Fisher estimation.
    # (Does not change grad-enabled state.)
    model.train()

    for batch in batches:
        if n_ok >= int(max_batches):
            break

        if begin_step_fn is not None:
            begin_step_fn()

        try:
            model.zero_grad(set_to_none=True)

            # Make absolutely sure autograd is enabled in case the caller is in a no_grad context.
            with torch.enable_grad():
                loss = loss_fn(model, batch)

            if not torch.is_tensor(loss):
                # allow loss_fn to return python floats (shouldn't happen, but be defensive)
                loss = torch.as_tensor(loss, device=next(model.parameters()).device)

            # If the loss isn't connected to any parameters, backward() will error.
            # This can happen if someone accidentally disabled requires_grad or computed loss under no_grad.
            if not loss.requires_grad:
                n_skipped_no_grad += 1
                continue

            loss.backward()

            # Accumulate squared grads for selected parameters.
            for name, p in model.named_parameters():
                if filter_fn is not None and not filter_fn(name):
                    continue
                if p.grad is None:
                    continue
                # Meta params can't be used here; skip (out-of-core expert not materialized).
                if p.device.type == "meta":
                    continue

                g2 = p.grad.detach().float().cpu().pow(2)
                if name not in fisher:
                    fisher[name] = torch.zeros_like(g2, device="cpu")
                fisher[name] += g2

            n_ok += 1

        finally:
            # IMPORTANT: only evict after we've copied grads to CPU.
            if end_step_fn is not None:
                end_step_fn()

    # Normalize by number of successful batches.
    if n_ok > 0:
        inv = 1.0 / float(n_ok)
        for k in list(fisher.keys()):
            fisher[k].mul_(inv)

    # If everything was skipped, emit a single-line warning for debugging.
    if n_ok == 0 and n_skipped_no_grad > 0:
        # Avoid noisy prints; but one message helps users find the issue.
        print(
            f"[estimate_fisher_diag] WARNING: skipped {n_skipped_no_grad} fisher batches "
            f"because loss.requires_grad was False. Returning empty fisher."
        )

    return fisher


def make_snapshot(
    model: nn.Module,
    *,
    name: str,
    fisher_diag: Dict[str, torch.Tensor],
    filter_fn: FilterFn = None,
) -> TaskSnapshot:
    """Create a snapshot of current parameters (theta*) and fisher diag.

    Stored tensors are on CPU for portability. Meta parameters are skipped.
    """
    params_star: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        if filter_fn is not None and not filter_fn(n):
            continue
        if p.device.type == "meta":
            continue
        params_star[n] = p.detach().float().cpu().clone()

    # Ensure fisher tensors are CPU float
    fisher_cpu: Dict[str, torch.Tensor] = {}
    for n, f in fisher_diag.items():
        if not torch.is_tensor(f):
            continue
        fisher_cpu[n] = f.detach().float().cpu()

    return TaskSnapshot(name=name, params_star=params_star, fisher_diag=fisher_cpu)


class EWC:
    def __init__(self, lambda_: float = 0.0):
        self.lambda_ = float(lambda_)
        self.tasks: list[TaskSnapshot] = []

    def add_task_snapshot(self, snapshot: TaskSnapshot) -> None:
        self.tasks.append(snapshot)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC quadratic penalty on the current model parameters.

        Out-of-core safety:
        - Skip any parameters currently on 'meta' (not materialized).
        - Only penalize parameters present in each task snapshot.
        """
        if self.lambda_ <= 0.0 or not self.tasks:
            # return a tensor on the model device to avoid device mismatch when added to loss
            dev = next(model.parameters()).device
            return torch.zeros((), device=dev)

        # We'll accumulate on the device of the current params (typically GPU).
        total = None
        for name, p in model.named_parameters():
            if p.device.type == "meta":
                continue
            # Only penalize if any task has fisher for this param.
            # We gather task terms lazily to avoid repeated dict lookups.
            term = None
            for task in self.tasks:
                f = task.fisher_diag.get(name, None)
                s = task.params_star.get(name, None)
                if f is None or s is None:
                    continue
                # Move fisher and star to param device on demand.
                f_d = f.to(device=p.device, non_blocking=True)
                s_d = s.to(device=p.device, non_blocking=True)
                t = (f_d * (p - s_d).pow(2)).sum()
                term = t if term is None else (term + t)

            if term is None:
                continue
            total = term if total is None else (total + term)

        if total is None:
            dev = next(model.parameters()).device
            return torch.zeros((), device=dev)

        return 0.5 * self.lambda_ * total

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch


@dataclass
class EWCTaskSnapshot:
    name: str
    params_star: Dict[str, torch.Tensor]  # CPU tensors
    fisher_diag: Dict[str, torch.Tensor]  # CPU tensors


class EWC:
    """Elastic Weight Consolidation (diagonal Fisher approximation).

    We store per-task:
      - theta* (parameter values after training on task)
      - diag Fisher estimate (importance)

    During subsequent training, we add a quadratic penalty.
    """

    def __init__(self, lambda_: float = 0.4):
        self.lambda_ = float(lambda_)
        self.tasks: List[EWCTaskSnapshot] = []

    def add_task_snapshot(self, snapshot: EWCTaskSnapshot) -> None:
        self.tasks.append(snapshot)

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        if not self.tasks or self.lambda_ <= 0:
            return torch.zeros((), device=next(model.parameters()).device)

        penalty = torch.zeros((), device=next(model.parameters()).device)
        named_params = dict(model.named_parameters())

        for task in self.tasks:
            for name, p_star in task.params_star.items():
                if name not in named_params:
                    continue
                p = named_params[name]
                f = task.fisher_diag.get(name)
                if f is None:
                    continue
                # move to device lazily
                f_dev = f.to(p.device)
                p_star_dev = p_star.to(p.device)
                penalty = penalty + (f_dev * (p - p_star_dev).pow(2)).mean()

        return self.lambda_ * penalty


@torch.no_grad()
def _clone_params(named_params: Dict[str, torch.nn.Parameter], *, filter_fn: Optional[Callable[[str], bool]] = None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for n, p in named_params.items():
        if filter_fn is not None and not filter_fn(n):
            continue
        out[n] = p.detach().cpu().clone()
    return out


def estimate_fisher_diag(
    model: torch.nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    loss_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    *,
    filter_fn: Optional[Callable[[str], bool]] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher as average squared gradients of `loss_fn`.

    This is a pragmatic approximation used widely in EWC-style implementations.
    """
    model.train()

    # Initialize accumulators
    fisher: Dict[str, torch.Tensor] = {}
    named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in named_params.items():
        if filter_fn is not None and not filter_fn(n):
            continue
        fisher[n] = torch.zeros_like(p, device=p.device)

    n_batches = 0
    for batch in batches:
        if max_batches is not None and n_batches >= max_batches:
            break

        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()

        for n, p in named_params.items():
            if n not in fisher:
                continue
            if p.grad is None:
                continue
            fisher[n] += p.grad.detach().pow(2)

        n_batches += 1

    if n_batches == 0:
        return {k: v.detach().cpu() for k, v in fisher.items()}

    for k in fisher:
        fisher[k] = (fisher[k] / float(n_batches)).detach().cpu()

    return fisher


def make_snapshot(
    model: torch.nn.Module,
    name: str,
    fisher_diag: Dict[str, torch.Tensor],
    *,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> EWCTaskSnapshot:
    params_star = _clone_params(dict(model.named_parameters()), filter_fn=filter_fn)
    # Ensure fisher keys match filter
    if filter_fn is not None:
        fisher_diag = {k: v for k, v in fisher_diag.items() if filter_fn(k)}
    return EWCTaskSnapshot(name=name, params_star=params_star, fisher_diag=fisher_diag)

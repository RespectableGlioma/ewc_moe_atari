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

    NOTE: This scaffold explicitly skips parameters currently on the 'meta' device.
    That matters for our out-of-core expert store, where cold experts are moved to meta
    after being written to disk.
    """

    def __init__(self, lambda_: float = 0.4):
        self.lambda_ = float(lambda_)
        self.tasks: List[EWCTaskSnapshot] = []

    def add_task_snapshot(self, snapshot: EWCTaskSnapshot) -> None:
        self.tasks.append(snapshot)

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        if not self.tasks or self.lambda_ <= 0:
            # If model has no parameters (shouldn't), just return 0.
            try:
                dev = next(model.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
            return torch.zeros((), device=dev)

        dev = next(model.parameters()).device
        penalty = torch.zeros((), device=dev)
        named_params = dict(model.named_parameters())

        for task in self.tasks:
            for name, p_star in task.params_star.items():
                p = named_params.get(name)
                if p is None:
                    continue
                if p.device.type == "meta":
                    # Cold expert currently offloaded; it's not being trained, so we don't need a penalty.
                    continue
                f = task.fisher_diag.get(name)
                if f is None:
                    continue
                # move to device lazily
                f_dev = f.to(p.device)
                p_star_dev = p_star.to(p.device)
                penalty = penalty + (f_dev * (p - p_star_dev).pow(2)).mean()

        return self.lambda_ * penalty


@torch.no_grad()
def _clone_params(
    named_params: Dict[str, torch.nn.Parameter],
    *,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for n, p in named_params.items():
        if filter_fn is not None and not filter_fn(n):
            continue
        if p.device.type == "meta":
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
    begin_step_fn: Optional[Callable[[], None]] = None,
    end_step_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher as average squared gradients of `loss_fn`.

    Pragmatic approximation used widely in EWC implementations.

    Out-of-core details:
      - Parameters may migrate across devices (CPU<->GPU) due to paging.
      - We accumulate Fisher on CPU, and always move grads to CPU before accumulating.
      - If begin_step_fn/end_step_fn are provided (ExpertStore pin scopes), end_step_fn
        is called AFTER gradients are read/accumulated to avoid moving parameters
        (and leaving grads behind) before we read them.
      - Parameters currently on the 'meta' device are skipped.
    """
    model.train()

    fisher: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for batch in batches:
        if max_batches is not None and n_batches >= max_batches:
            break

        if begin_step_fn is not None:
            begin_step_fn()
        try:
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            loss.backward()

            # Accumulate squared grads. Always accumulate on CPU so we don't care
            # where parameters (or grads) live.
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.device.type == "meta":
                    continue
                if filter_fn is not None and not filter_fn(n):
                    continue
                if p.grad is None:
                    continue

                g = p.grad.detach()
                if g.is_sparse:
                    g = g.to_dense()
                g2_cpu = g.float().cpu().pow(2)

                if n not in fisher:
                    fisher[n] = torch.zeros_like(g2_cpu, device="cpu")
                fisher[n] += g2_cpu

        finally:
            # IMPORTANT: end_step_fn may evict experts / move parameters. Do it only
            # after we've read grads into CPU buffers.
            if end_step_fn is not None:
                end_step_fn()

        n_batches += 1

    if n_batches == 0:
        return {k: v.detach().cpu() for k, v in fisher.items()}

    for k in list(fisher.keys()):
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
    if filter_fn is not None:
        fisher_diag = {k: v for k, v in fisher_diag.items() if filter_fn(k)}
    return EWCTaskSnapshot(name=name, params_star=params_star, fisher_diag=fisher_diag)

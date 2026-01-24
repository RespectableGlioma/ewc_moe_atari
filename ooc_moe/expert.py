
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ExpertOptState:
    """Per-expert optimizer state stored on CPU (and optionally persisted to disk)."""

    kind: str  # "sgd", "sgd_momentum", "adamw"
    step: int = 0
    tensors: TensorDict = field(default_factory=dict)


@dataclass
class ExpertState:
    """
    Canonical expert state on CPU (or persisted to disk).

    params: float32 tensors on CPU (master weights)
    opt: optional optimizer state (e.g., momentum buffers) on CPU
    """
    params: TensorDict
    opt: Optional[ExpertOptState] = None


def _activation(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(x)
    if kind == "gelu":
        return F.gelu(x)
    if kind == "silu":
        return F.silu(x)
    raise ValueError(f"Unknown activation: {kind}")


def make_expert_state(
    d_model: int,
    d_hidden: int,
    *,
    seed: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = False,
    init_scale: float = 0.02,
    with_momentum: bool = False,
) -> ExpertState:
    """
    Create a single expert's weights (and optional momentum) on CPU.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Use a simple FFN: y = W2(act(W1 x + b1)) + b2
    w1 = torch.randn(d_hidden, d_model, generator=g, device=device, dtype=dtype) * init_scale
    b1 = torch.zeros(d_hidden, device=device, dtype=dtype)
    w2 = torch.randn(d_model, d_hidden, generator=g, device=device, dtype=dtype) * init_scale
    b2 = torch.zeros(d_model, device=device, dtype=dtype)

    params: TensorDict = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    if pin_memory and device == "cpu" and torch.cuda.is_available():
        for k in list(params.keys()):
            params[k] = params[k].pin_memory()

    opt: Optional[ExpertOptState] = None
    if with_momentum:
        tensors = {f"m_{k}": torch.zeros_like(v) for k, v in params.items()}
        if pin_memory and device == "cpu" and torch.cuda.is_available():
            for k in list(tensors.keys()):
                tensors[k] = tensors[k].pin_memory()
        opt = ExpertOptState(kind="sgd_momentum", step=0, tensors=tensors)

    return ExpertState(params=params, opt=opt)


def expert_forward(
    x: torch.Tensor,
    params: TensorDict,
    *,
    activation: str = "gelu",
) -> torch.Tensor:
    """
    Functional expert forward.
    """
    h = F.linear(x, params["w1"], params["b1"])
    h = _activation(h, activation)
    y = F.linear(h, params["w2"], params["b2"])
    return y


class ExpertSlot:
    """
    A GPU-resident slot that can hold one expert's parameters.

    This is the "HBM cache line" for experts.
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        *,
        device: torch.device,
        compute_dtype: torch.dtype,
        activation: str = "gelu",
    ):
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.device = device
        self.compute_dtype = compute_dtype
        self.activation = activation

        # Leaf tensors that will receive gradients.
        self.params: TensorDict = {
            "w1": torch.empty(d_hidden, d_model, device=device, dtype=compute_dtype, requires_grad=True),
            "b1": torch.empty(d_hidden, device=device, dtype=compute_dtype, requires_grad=True),
            "w2": torch.empty(d_model, d_hidden, device=device, dtype=compute_dtype, requires_grad=True),
            "b2": torch.empty(d_model, device=device, dtype=compute_dtype, requires_grad=True),
        }

    def load_from_cpu(self, cpu_state: ExpertState, non_blocking: bool = True) -> int:
        """
        Copy CPU master weights into this slot. Returns bytes moved host->device.
        """
        nbytes = 0
        with torch.no_grad():
            for k, t in self.params.items():
                src = cpu_state.params[k]
                # Cast to compute dtype on transfer.
                t.copy_(src.to(device=self.device, dtype=self.compute_dtype, non_blocking=non_blocking))
                nbytes += src.numel() * src.element_size()
        # Clear any stale grads
        self.zero_grad()
        return nbytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return expert_forward(x, self.params, activation=self.activation)

    def zero_grad(self) -> None:
        for t in self.params.values():
            t.grad = None

    def grads_to_cpu(self) -> TensorDict:
        """
        Extract gradients as CPU float32 tensors.
        """
        grads: TensorDict = {}
        for k, t in self.params.items():
            if t.grad is None:
                raise RuntimeError(f"Missing grad for {k}. Did you call backward?")
            grads[k] = t.grad.detach().to(device="cpu", dtype=torch.float32)
        return grads

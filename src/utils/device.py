from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Prefer CUDA if available.

    NOTE: torch.device('cuda') is *not* equal to torch.device('cuda:0').
    Some parts of this repo compare devices for cache bookkeeping (HBM hit/miss).
    We therefore return an explicit CUDA index when CUDA is available.
    """
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")

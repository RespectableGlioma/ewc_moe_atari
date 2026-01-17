from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Prefer CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

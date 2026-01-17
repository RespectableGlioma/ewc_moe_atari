from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seeds(seed: int, *, deterministic_torch: bool = False) -> None:
    """Set RNG seeds for (most) common sources of randomness."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch isn't a hard dependency for seed setting.
        pass


def maybe_seed_env(env, seed: Optional[int]) -> None:
    """Best-effort environment seeding, compatible with Gymnasium."""
    if seed is None:
        return

    try:
        env.reset(seed=seed)
    except TypeError:
        # Some envs may not support reset(seed=...).
        try:
            env.seed(seed)
        except Exception:
            pass

    try:
        env.action_space.seed(seed)
    except Exception:
        pass

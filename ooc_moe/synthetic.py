
from __future__ import annotations

from typing import Tuple

import torch


class EnvEpisodeSampler:
    """
    Generates a stream of environment IDs with controllable locality.

    - episode_len: number of consecutive steps with the same env id
    """
    def __init__(self, n_envs: int, episode_len: int, seed: int = 0):
        self.n_envs = n_envs
        self.episode_len = max(1, int(episode_len))
        self.g = torch.Generator(device="cpu")
        self.g.manual_seed(seed)
        self._t = 0
        self._cur = int(torch.randint(0, n_envs, (1,), generator=self.g).item())

    def next(self) -> int:
        if self._t % self.episode_len == 0:
            self._cur = int(torch.randint(0, self.n_envs, (1,), generator=self.g).item())
        self._t += 1
        return self._cur


class EnvMarkovSampler:
    """
    Markov env stream: with prob p_stay keep same env, else switch.
    """
    def __init__(self, n_envs: int, p_stay: float, seed: int = 0):
        self.n_envs = n_envs
        self.p_stay = float(p_stay)
        self.g = torch.Generator(device="cpu")
        self.g.manual_seed(seed)
        self._cur = int(torch.randint(0, n_envs, (1,), generator=self.g).item())

    def next(self) -> int:
        u = torch.rand((), generator=self.g).item()
        if u > self.p_stay:
            self._cur = int(torch.randint(0, self.n_envs, (1,), generator=self.g).item())
        return self._cur


class SyntheticEnvTeacher:
    """
    A fixed, environment-specific linear mapping y = x @ A_env + noise.

    Supports both:
      - a single env_id for the whole batch
      - per-token env_ids (shape [B])
    """
    def __init__(self, n_envs: int, d_model: int, seed: int = 0, noise_std: float = 0.01):
        self.n_envs = n_envs
        self.d_model = d_model
        self.noise_std = float(noise_std)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        # Teacher matrices: [n_envs, d, d]
        self.A = torch.randn(n_envs, d_model, d_model, generator=g) / (d_model ** 0.5)

    def sample(self, batch_size: int, env_id: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, self.d_model, device=device, dtype=dtype)
        A = self.A[env_id].to(device=device, dtype=dtype)
        y = x @ A
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return x, y

    def sample_per_token(self, env_ids: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        env_ids: [B] int64 on any device
        """
        if env_ids.dtype != torch.long:
            env_ids = env_ids.long()
        B = env_ids.shape[0]
        x = torch.randn(B, self.d_model, device=device, dtype=dtype)
        A_batch = self.A.index_select(0, env_ids.to("cpu")).to(device=device, dtype=dtype)  # [B, d, d]
        y = torch.einsum("bd,bde->be", x, A_batch)
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return x, y

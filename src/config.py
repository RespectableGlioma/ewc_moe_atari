from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    frame_stack: int = 4
    reward_clip: bool = True

    # MoE model
    num_experts: int = 32
    router_hidden_dim: int = 128
    expert_hidden_dim: int = 256
    feature_dim: int = 512
    expert_top_k: int = 2  # sparse top-k experts per step
    load_balance_weight: float = 0.01
    router_noise: float = 1.0

    # Expert Store (HBM/DRAM/NVMe)
    hbm_expert_capacity: int = 4          # how many experts may be resident on GPU
    dram_expert_capacity: int = 16        # how many experts may be resident on CPU before spilling to disk
    enable_nvme_tier: bool = True         # if True, evicted CPU experts are written to disk and moved to meta
    pin_cpu_memory: bool = True           # enables non_blocking H2D copies

    # DQN / training
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 16
    seq_len: int = 8

    # Exploration (epsilon-greedy)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 50_000

    # Target network
    target_update_interval: int = 1000

    # Day/Night schedule
    games_per_day: int = 5
    day_steps_per_game: int = 6000
    sleep_updates_per_game: int = 500

    # Salience
    salience_alpha: float = 0.6  # sampling exponent
    td_error_weight: float = 1.0
    policy_surprisal_weight: float = 0.2
    softmax_temp_for_surprisal: float = 1.0

    # EWC
    ewc_lambda: float = 0.4
    fisher_batches: int = 25
    top_experts_per_game: int = 4
    protect_encoder: bool = False  # include encoder params in EWC
    protect_experts: bool = True   # include flagged experts in EWC

    # Logging / viz
    log_every_sleep_steps: int = 50

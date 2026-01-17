# EWC + MoE Router (RNN) for Multi-Atari Day/Night Training (Gymnasium)

This is a **research scaffold** you can run in **Google Colab** to test the idea:

- **Day (acting):** play a small set of Atari games sequentially (correlated stream), collect trajectories and salience.
- **Sleep (learning):** replay only the games encountered during that day, but **shuffle** to produce more IID-like updates.
- **Continual retention:** use **Elastic Weight Consolidation (EWC)** on shared parameters (router / trunk and optionally experts).
- **Sparsity/modularity:** a **GRU router** gates a set of **experts** (MoE) producing Q-values.

The code is intentionally modular so you can later:
- swap the caching simulator for real HBM/DRAM/NVMe swapping,
- plug in a better learner (IMPALA/V-trace, R2D2, etc.),
- scale expert counts.

## Key compatibility details

- Uses **Gymnasium** and Atari environments (ALE).
- Uses `FrameStackObservation` (Gymnasium v1.0+), not `FrameStack`.
- Uses `full_action_space=True` so each Atari game exposes **Discrete(18)** actions (unified action space).
- Clips rewards to `[-1, 1]`.

## Project layout

- `EWC_MoE_Atari.ipynb` — the Colab-friendly entry point
- `src/` — importable python package with env building, model, replay, and training

## Notes

This is a *prototype*. Training Atari well takes lots of compute. The notebook defaults to a **smoke test** run (few thousand steps) to verify the plumbing.

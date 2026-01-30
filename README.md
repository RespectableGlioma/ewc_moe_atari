# Out-of-Core MOE for Multi-Task RL

A prototype for training Mixture of Experts models across memory hierarchies (HBM/DRAM/NVMe) with environment-aware expert caching, applied to sequential multi-task Atari learning.

## Core Hypothesis

**Sequential training normally causes catastrophic forgetting** because the same parameters get overwritten for each new task. But with MOE + environment-aware routing + tiered storage:

1. **Different games activate different experts** → natural task separation
2. **Game identity is temporally correlated** → perfect for caching (you play Breakout for 1000s of steps)
3. **Sequential presentation becomes a feature** → cold experts stay cold and protected, hot experts specialize

We're betting that **sparse routing + memory hierarchy = implicit continual learning**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVATION ENCODER                          │
│    (CNN for Atari frames, always in HBM - shared across games)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT DETECTOR                          │
│    (Predicts which game → prefetches relevant experts)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMER BACKBONE (MOE FFN Layers)               │
│                                                                  │
│   Experts stored across:                                         │
│   • HBM (32 experts) - hot, frequently used                     │
│   • DRAM (128 experts) - warm, recently used                    │
│   • NVMe (all 256+) - cold storage                              │
│                                                                  │
│   Prefetching hides storage latency when game is detected       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACTION HEAD + VALUE HEAD                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. TieredExpertStore (`core/tiered_store.py`)
Manages expert parameters across HBM/DRAM/NVMe with:
- LRU caching per tier
- Asynchronous prefetching
- Access pattern tracking
- Gradient accumulation for cold experts

### 2. MOE Layers (`core/moe_layers.py`)
- Router network (always in HBM)
- Top-k expert selection
- Load balancing loss
- Integration with tiered storage

### 3. Environment Detector (`core/env_detector.py`)
Lightweight CNN that:
- Classifies current game from observations
- Predicts which experts will be needed
- Issues prefetch commands ahead of computation

### 4. MoE RL Agent (`models/moe_agent.py`)
Full Decision Transformer-style architecture with:
- Shared observation encoder
- MOE transformer blocks
- Action and value heads

## Installation

```bash
# Clone and setup
git clone <repo>
cd ooc_moe

# Install dependencies
pip install -r requirements.txt

# For Atari support
pip install gymnasium[atari] ale-py
```

## Usage

### Run Tests
```bash
python run.py --mode test
```

### Train on Dummy Environments
```bash
python run.py --mode train \
    --num_experts 256 \
    --hbm_capacity 32 \
    --dram_capacity 128 \
    --num_games 5 \
    --steps_per_game 100000
```

### Train on Atari
```bash
python run.py --mode train_atari \
    --num_experts 256 \
    --num_games 10 \
    --steps_per_game 1000000
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_experts` | 256 | Total number of experts |
| `--expert_dim` | 512 | Expert input/output dimension |
| `--expert_hidden_dim` | 2048 | Expert FFN hidden dimension |
| `--num_layers` | 6 | Transformer layers |
| `--top_k` | 2 | Experts per token |
| `--hbm_capacity` | 32 | Experts in HBM (GPU) |
| `--dram_capacity` | 128 | Experts in DRAM (CPU) |
| `--context_len` | 32 | Transformer context window |

## Key Metrics

The system tracks:

1. **Cache Hit Rate**: How often needed experts are already in HBM
2. **Prefetch Effectiveness**: % of hits that were prefetched
3. **Expert Specialization**: Mutual information between games and experts
4. **Retention**: Performance on old games after training on new ones

## Expected Results

If the hypothesis is correct:

1. **High cache hit rates** (>80%) during gameplay due to temporal correlation
2. **Emergent expert specialization** - different experts activate for different games
3. **Reduced catastrophic forgetting** - old game performance retained
4. **Efficient memory usage** - only active experts in fast storage

## Project Structure

```
ooc_moe/
├── core/
│   ├── tiered_store.py    # Memory hierarchy management
│   ├── moe_layers.py      # MOE layer implementations
│   └── env_detector.py    # Environment detection & prefetching
├── models/
│   └── moe_agent.py       # Full RL agent architecture
├── training/
│   └── ppo_trainer.py     # PPO + sequential game training
├── envs/
│   └── atari_wrappers.py  # Atari environment wrappers
├── tests/
│   └── test_harness.py    # Validation tests
├── run.py                 # Main entry point
└── requirements.txt
```

## Extending the System

### Adding Real NVMe Storage

Replace the simulated storage in `TieredExpertStore`:

```python
# Use memory-mapped files for true out-of-core
self.nvme_store = np.memmap('experts.dat', dtype='float32', mode='r+',
                            shape=(num_experts, expert_size))
```

### Multi-Node Training

The tiered storage abstraction supports distributed training:
- Each node manages its own HBM/DRAM cache
- NVMe storage can be shared via NFS or object storage
- Gradient accumulation handles asynchronous updates

### Alternative Training Methods

The architecture supports exploration of non-backprop methods:
- Forward-Forward algorithm
- Hebbian learning for expert updates
- Local learning rules

## Connection to Causal Modeling

The Environment Detector learns a **causal model of expert activation**:
- What input features predict expert needs
- Temporal dependencies in expert access
- Compositional patterns across games

This connects to your broader research on causal world models:
- Rules emerge as "equivalence classes" of expert routing patterns
- Deterministic aspects (game identity → expert cohort) vs stochastic (moment-to-moment routing)

## Citation

If you use this code, please cite:
```
@misc{ooc_moe_2025,
  title={Out-of-Core Mixture of Experts for Multi-Task Reinforcement Learning},
  author={EKO},
  year={2025}
}
```

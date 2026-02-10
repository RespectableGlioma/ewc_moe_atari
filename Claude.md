# Out-of-Core Mixture of Experts for Continual Atari Learning

## Project Overview

This project explores **continual learning** across multiple Atari games using a hierarchical Mixture of Experts (MoE) architecture. The core idea: instead of training one monolithic network on all games (which suffers from catastrophic forgetting) or separate networks per game (which doesn't scale), we use a small meta-agent to dynamically route observations to specialized experts stored on disk.

### The Continual Learning Problem

When neural networks learn task B after task A, they typically forget task A — this is **catastrophic forgetting**. Standard approaches:
- **Regularization** (EWC, SI): Penalize changes to important weights → limited capacity
- **Replay buffers**: Store old examples → memory overhead, privacy concerns
- **Architecture growth**: Add capacity per task → unbounded growth

### Our Approach: Hierarchical Out-of-Core MoE

We separate **what to do** (experts) from **when to do it** (meta-agent):

```
┌─────────────────────────────────────────┐
│  Meta-Agent (GPU) - Always Resident     │
│  • Encodes observations to game codes   │
│  • Detects game switches via KL spike   │
│  • Routes to appropriate expert         │
└──────────────────┬──────────────────────┘
                   │ selects
                   ▼
┌─────────────────────────────────────────┐
│  Active Expert (GPU) - Swappable        │
│  • Full actor-critic network            │
│  • Trained with PPO on current game     │
└──────────────────┬──────────────────────┘
                   │ swap in/out
                   ▼
┌─────────────────────────────────────────┐
│  Expert Library (Disk/NVMe)             │
│  • expert_0000.pt, expert_0001.pt, ...  │
│  • Indexed by learned VQ codes          │
└─────────────────────────────────────────┘
```

**Key insight**: The meta-agent is tiny (~1M params) and learns game-identity, not game-playing. Experts are large (~2M params each) but only one is in GPU memory at a time. This allows scaling to many games without proportional GPU memory growth.

---

## Architecture

### Meta-Agent (MetaRSSM)

A Recurrent State-Space Model that compresses observations into discrete game codes:

```
obs(t) → CNN Encoder → e(t)
                         ↓
           GRU(e(t), z(t-1), h(t-1)) → h(t)
                                        ↓
                    Posterior: q(z|h,e) → VQ Quantize → z(t), code_idx
                    Prior:     p(z|h)   → distribution over codes
```

**Components:**
- **CNN Encoder**: Nature DQN architecture, extracts visual features
- **GRU**: Maintains temporal context across observations
- **VQ Codebook**: 64 discrete "game prototype" embeddings
- **Prior Network**: Predicts code from dynamics alone (for REINFORCE)
- **Posterior Network**: Infers code given observation (for VQ selection)

**Switch Detection**: KL divergence between prior and posterior spikes when the game changes — the prior (based on dynamics) expects one game, but the posterior (seeing the observation) infers a different one.

### Experts

Standard actor-critic networks with Nature CNN backbone:

- **Backbone**: 3-layer CNN → 512-dim features
- **Actor**: MLP → policy logits over unified action space
- **Critic**: MLP → value estimate
- **Metadata**: Tracks games trained, total frames, best reward

### Expert Manager

Orchestrates expert lifecycle with two selection mechanisms:

1. **Reward-Weighted Affinity**: Learned code→expert mappings based on performance
   ```python
   score = ema_reward * (1 + log(visit_count + 1) * stickiness)
   ```

2. **Embedding Similarity**: Fallback using cosine similarity to expert centroids

### Unified Action Space

Different Atari games have different action sets. We build a union:

| Unified Idx | Semantic    | Breakout | Pong | SpaceInvaders |
|-------------|-------------|----------|------|---------------|
| 0           | NOOP        | ✓        | ✓    | ✓             |
| 1           | FIRE        | ✓        | ✓    | ✓             |
| 2           | RIGHT       | ✓        | ✓    | ✓             |
| 3           | LEFT        | ✓        | ✓    | ✓             |
| 4           | RIGHTFIRE   | ✗        | ✓    | ✓             |
| 5           | LEFTFIRE    | ✗        | ✓    | ✓             |

Per-game masks set invalid logits to `-inf` before sampling.

---

## Training Loop: Day/Night Cycle

### Day Phase (Expert Training)
```python
for game in curriculum.sample(K):
    obs = env.reset()

    # Meta-agent encodes observation
    meta_state, outputs = meta_agent(obs, meta_state)
    code_idx = meta_agent.get_game_code(meta_state)

    # Store (h, code) for REINFORCE
    selection_data.append((meta_state.h.detach(), code_idx.detach()))

    # Retrieve expert via affinity or similarity
    expert = expert_manager.retrieve_or_create(embedding, code_idx)

    # Train expert with PPO
    for _ in range(updates_per_game):
        obs, metrics = ppo_trainer.step(env, obs, game_name)

    # Update affinity based on game reward
    expert_manager.update_affinity(game_reward, code_idx)

    # Meta-agent unsupervised update (KL + VQ + transition prediction)
    meta_loss = meta_agent.compute_loss(trajectory)
    meta_loss.backward()
```

### Night Phase (Meta-Agent REINFORCE)
```python
# Recompute log_probs with current parameters (avoid stale gradients)
for h_state, code_idx in selection_data:
    prior_logits = meta_agent.prior_net(h_state)
    log_prob = F.log_softmax(prior_logits, dim=-1)
    log_probs.append(log_prob.gather(1, code_idx))

# REINFORCE: encourage selections that led to high reward
advantage = cumulative_reward - baseline
reinforce_loss = -advantage * sum(log_probs)
reinforce_loss.backward()
```

---

## What Was Built

### Core Modules (`core/`)

| File | Purpose |
|------|---------|
| `meta_rssm.py` | MetaRSSM with VQ codebook, KL switch detection, REINFORCE support |
| `expert.py` | Actor-critic expert with unified action masking |
| `expert_manager.py` | Expert lifecycle, affinity-based selection, pruning |
| `tiered_store.py` | Disk storage with CPU cache and async loading |

### Training (`training/`)

| File | Purpose |
|------|---------|
| `ppo_trainer.py` | PPO with GAE, unified action space integration |
| `rollout_buffer.py` | Experience storage for PPO updates |

### Environments (`envs/`)

| File | Purpose |
|------|---------|
| `atari_wrappers.py` | Standard Atari preprocessing (frame stack, resize, etc.) |
| `game_curriculum.py` | Random/Markov/Periodic game scheduling |
| `action_space.py` | UnifiedActionSpace with per-game masks |

### Entry Points

| File | Purpose |
|------|---------|
| `train.py` | Full training script with CLI args |
| `ooc_moe_colab.ipynb` | Interactive notebook for Colab with checkpointing |

---

## Training Results

### 300-Day Run (8 Games, 19.2M Frames)

```
Total experts: 20
Affinity selections: 1322 (89%)
Embedding selections: 158 (11%)

Top Expert: expert_0015
  - 14.8M frames (77% of all training)
  - Trained on: Frostbite, Pong, Seaquest, BeamRider, Breakout, Qbert, SpaceInvaders, Enduro
```

**Observations:**
1. **Affinity dominates**: System learned to use reward-based routing over similarity
2. **Mode collapse**: Meta-agent converged to routing most observations to one expert
3. **Pruning works**: Reduced from 31 to 20 experts during training
4. **REINFORCE converged**: Mean log prob reached -0.0038 (very confident selections)

### Known Issues

1. **Extreme concentration**: One expert receives most training, others starve
2. **Undertrained experts**: 8 experts with only 12.8K frames (1 game visit)
3. **Pong stuck at -21**: Sparse reward + long horizon makes learning difficult

---

## Implementation Details

### Avoiding Stale Gradients

During day phase, we update RSSM parameters after computing selection log_probs. By night, the gradient graph is stale.

**Solution**: Store `(h_state.detach(), code_idx)` during day, recompute log_probs at night with current parameters.

### Affinity-Based Expert Selection

```python
class CodeExpertAffinity:
    def get_best_expert(self, code_idx):
        for expert_id, aff in self.affinity[code_idx].items():
            stickiness = log(aff['visit_count'] + 1) * stickiness_scale
            score = aff['ema_reward'] * (1 + stickiness)

        # 10% exploration: fall back to embedding similarity
        if random() < 0.1:
            return None

        return best_expert

    def update(self, code_idx, expert_id, reward):
        aff['ema_reward'] = 0.9 * aff['ema_reward'] + 0.1 * reward
        aff['visit_count'] += 1
```

### Expert Pruning

```python
def prune_experts(min_frames=25000, min_affinity_score=0.0, dry_run=True):
    # Keep if: frames >= threshold OR affinity >= threshold
    # Protected: active/prefetched experts
    # On delete: redistribute affinity to similar remaining expert
```

---

## Future Directions

### 1. Address Mode Collapse

The meta-agent learned to route everything to one expert. Options:
- **Entropy bonus** in REINFORCE to encourage diverse selections
- **Annealed exploration** in affinity (start high, decay over training)
- **Per-game routing constraints** ensuring each game has dedicated expert capacity

### 2. Gumbel-Softmax for End-to-End Gradients

Replace VQ hard selection with differentiable relaxation:
```python
y_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
y_hard = one_hot(y_soft.argmax()) - y_soft.detach() + y_soft  # Straight-through
```
This would allow gradients to flow from expert performance through code selection.

### 3. Expert Merging

Instead of pruning, merge similar undertrained experts:
- Average weights of experts with similar centroids
- Combine affinity mappings
- Reduces fragmentation while preserving learned representations

### 4. Hierarchical Experts

Add structure within experts:
- Shared backbone across all experts (lower layers)
- Expert-specific heads (upper layers)
- Reduces total parameters while maintaining specialization

### 5. Curriculum Learning

Current random/Markov curriculum may not be optimal:
- **Self-paced**: Focus on games where expert is improving
- **Adversarial**: Focus on games where expert is struggling
- **Balanced**: Ensure minimum training per game

### 6. Evaluation Protocol

Need systematic evaluation:
- Per-game scores vs random policy baseline
- Transfer: Does training on game A help game B?
- Forgetting: After training game B, how much does game A degrade?

---

## File Structure

```
ewc_moe_atari/
├── core/
│   ├── __init__.py
│   ├── meta_rssm.py      # Meta-agent with VQ codebook
│   ├── expert.py         # Actor-critic expert
│   ├── expert_manager.py # Lifecycle + affinity selection
│   └── tiered_store.py   # Disk storage
├── training/
│   ├── __init__.py
│   ├── ppo_trainer.py    # PPO implementation
│   └── rollout_buffer.py # Experience buffer
├── envs/
│   ├── __init__.py
│   ├── atari_wrappers.py # Atari preprocessing
│   ├── game_curriculum.py # Game scheduling
│   └── action_space.py   # Unified action space
├── train.py              # CLI training script
├── ooc_moe_colab.ipynb   # Colab notebook
├── Claude.md             # This file
└── requirements.txt
```

---

## References

- **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning" (2017)
- **World Models**: Ha & Schmidhuber, "World Models" (2018)
- **Dreamer**: Hafner et al., "Dream to Control" (2019)
- **PPO**: Schulman et al., "Proximal Policy Optimization" (2017)
- **EWC**: Kirkpatrick et al., "Overcoming catastrophic forgetting" (2017)

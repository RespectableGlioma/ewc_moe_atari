Let's start this over. We want to enable out-of-core training of DNNs, and our inspiration is elastic weight consolidation. The task is sequential Atari games (like the original EWC paper). I htink we want to approach as a meta-learning problem. The outer loop meta-agent which is always in HBM3 predicts which experts on NVME need to be loaded into memory, play proceeds as normal, when the metagent detects the game has changed it writes all of the current expert to disk, and then loads up a new one. If none of the stored experts on disk are within some epsilon of similarity to the predict game, then it can also initialize a new agent. Something like this?

Yes, this is a much cleaner framing. You're essentially proposing a **hierarchical memory architecture** that mirrors the hardware hierarchy:

```
┌─────────────────────────────────────────┐
│  Meta-Agent (HBM3) - Always Resident    │
│  • Game detector / context encoder      │
│  • Expert similarity predictor          │
│  • Load/save orchestrator               │
└──────────────────┬──────────────────────┘
                   │ predicts & manages
                   ▼
┌─────────────────────────────────────────┐
│  Active Expert (HBM3) - Swappable       │
│  • Full game-playing network            │
│  • Currently loaded from NVMe           │
└──────────────────┬──────────────────────┘
                   │ swap in/out
                   ▼
┌─────────────────────────────────────────┐
│  Expert Library (NVMe)                  │
│  • Expert_0.pt, Expert_1.pt, ...        │
│  • Indexed by learned embeddings        │
└─────────────────────────────────────────┘
```

The key departures from vanilla EWC:

1. **No Fisher matrices** - Instead of protecting weights with quadratic penalties, you just... don't overwrite them. Complete isolation.

2. **Meta-learning the routing** - The outer loop learns *when* to switch and *what* to load, which is the actual hard problem EWC doesn't address.

3. **Explicit novelty detection** - The epsilon-similarity threshold gives you a principled way to say "this is a new game, allocate fresh capacity."

A few questions to sharpen the design:

- **What does the meta-agent see?** Raw frames? A frozen encoder's embeddings? Running statistics of observations/rewards?
- **How does it learn?** Does it get a training signal from expert performance, or is it purely unsupervised game-change detection?
- **Latency budget** - How many frames can you tolerate during a swap? This determines whether you need predictive prefetching or can react.

What's your intuition on the meta-agent architecture?

What the meta-agent sees is a hard thing... ideally it would be sets of Raw Frames or Raw Frames in some recurrent structure that get passed into an encoder. I feel like a Recurrent State Space Model might be particularly good at identifying game transitions?

The meta-agent's training signal should be in the overall expert performance reward after K-sets of games ... this formulation is nice because it permits a sort of day/night cycle to a day of k-games in the inner loop and then a "night-time" update of the meta-agent.

We could only predictive pre-fetch if there was some correlation between the ordering of the K-games. I like making this part of the environment to see if we can coax the meta-agent to learn these meta-trajectories across games. Let's try for this too?

This is beautiful. You've essentially created a **bi-level MDP** where the meta-agent is doing RL over game-sequences while experts do RL within games. The SSM insight is sharp—they're explicitly designed to compress long sequences into fixed-size states, which is exactly what you need for detecting distributional shifts.

Let me sketch the full picture:

```
┌─────────────────────────────────────────────────────────────┐
│                    META-TRAJECTORY LEVEL                     │
│  Game₁ ──→ Game₂ ──→ Game₃ ──→ ... ──→ Gameₖ               │
│    │         │         │                  │                  │
│    └─────────┴─────────┴──────────────────┘                  │
│         Meta-agent learns transition structure               │
│         Reward = Σᵢ ExpertPerformance(Gameᵢ)                │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ night update
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 META-AGENT (SSM in HBM3)                     │
│                                                              │
│   Frame ──→ [Encoder] ──→ [S4/Mamba Block] ──→ h_t          │
│                                   │                          │
│              ┌────────────────────┼────────────────┐         │
│              ▼                    ▼                ▼         │
│         z_t (game          p(next_game|h)     switch_t       │
│         embedding)         (prefetch dist)    (0/1)          │
└─────────────────────────────────────────────────────────────┘
```

**The three outputs are key:**

1. **z_t (game embedding)** — Matches against expert keys for retrieval. This is your similarity metric for the ε-threshold.

2. **p(next_game | h_t)** — If game orderings are correlated, this becomes non-uniform. The meta-agent can start prefetching Expert_j while Expert_i is still playing the final frames of Game_i.

3. **switch_t** — Binary detector for "the game just changed." Triggers the save-current / load-next cycle.

**The day/night cycle:**

```
DAY (inner loop, K games):
  for game in game_sequence:
      while not done:
          meta_agent.update_state(frame)
          if meta_agent.switch_detected():
              save(current_expert)
              next_expert = retrieve_or_create(meta_agent.z_t)
              load(next_expert)  # or use prefetched
          action = current_expert.act(frame)
          expert.train_step()  # standard RL
      
      cumulative_reward += episode_reward
      
NIGHT (outer loop):
  meta_agent.update(cumulative_reward)  # REINFORCE or similar
```

**The prefetching game:**

If you control the environment's game ordering, you can test increasingly structured curricula:

| Curriculum | Structure | Meta-agent learns... |
|------------|-----------|---------------------|
| Random | Uniform over games | Just detection, no prefetch benefit |
| Markov | P(g' \| g) non-uniform | First-order transition probs |
| Periodic | A→B→C→A→B→C... | Deterministic cycles |
| Contextual | Pattern depends on meta-state | Higher-order dependencies |

The Markov case is the sweet spot for demonstrating prefetch value—the meta-agent can learn that Breakout usually follows Pong and start loading the Breakout expert 100ms before the transition.

**One tension:** The SSM needs to run on every frame to maintain h_t, but you want it tiny enough to fit in HBM alongside the active expert. Probably fine—a small Mamba with a frozen conv encoder should be <100M params.

Want me to draft a concrete architecture spec with parameter counts and the training loop?

Ah, the RSSM from Dreamer—that's cleaner. You get game-change detection almost for free from the world model's prediction error. When the game switches, the learned dynamics suddenly fail and KL / reconstruction loss spikes.

```
┌─────────────────────────────────────────────────────────────┐
│                    META-AGENT (RSSM)                         │
│                                                              │
│   x_t ──→ [CNN Encoder] ──→ ẑ_t (posterior sample)          │
│                                    │                         │
│                                    ▼                         │
│   h_{t-1} ──→ [GRU] ──→ h_t ──→ [Prior] ──→ z_t (predicted) │
│                │                                             │
│                ▼                                             │
│   Outputs:                                                   │
│   • KL(ẑ_t || z_t) → switch signal (spike = game changed)   │
│   • e_t = MLP(h_t) → game embedding for retrieval           │
│   • p(g_{next}|h_t) → prefetch distribution                 │
└─────────────────────────────────────────────────────────────┘
```

**Why this works elegantly:**

| Signal | Source | Use |
|--------|--------|-----|
| Switch detection | KL divergence spike | Trigger save/load |
| Game identity | Discrete codes from VQVAE | Expert key matching |
| Prefetch hint | Learned transition model over game-codes | Async loading |

The discrete codebook is particularly nice—you can literally use the VQ indices as a hash for expert lookup. If the current observation quantizes to codes that historically belonged to "Breakout-like" games, you retrieve that expert.

**A100 80GB budget:**

```
Meta-agent RSSM:     ~100M params  →   400 MB
Active expert:       ~50M params   →   200 MB  
Prefetched expert:   ~50M params   →   200 MB
Replay buffer:                     → 10-20 GB
Optimizer states:                  →  2-3 GB
                                   ─────────
Headroom:                          → 50+ GB free
```

You could honestly hold 10+ experts in HBM simultaneously on an A100. The NVMe story becomes compelling when you're either:
1. Scaling to 50+ games (the full Atari-57 suite)
2. Using larger experts
3. Wanting to demonstrate the principle cleanly

**One architectural question:** In Dreamer, the RSSM is trained jointly with the actor-critic on world model rollouts. Here, the experts are separate and the meta-agent sees their rewards externally. Do you want the meta-agent to:

(a) **Learn its own world model** of the observation stream (unsupervised, just for detection/embedding)

(b) **Piggyback on expert world models** if experts are also Dreamer-style

(c) **Hybrid** — small meta-RSSM for detection, but can query expert world models for richer game-specific predictions

My instinct is (a) keeps things cleanest—the meta-agent's job is fundamentally different from playing the game.

Should I spec out the concrete encoder/GRU/codebook dimensions?

(a) exactly... the meta-agent learns its own more abstract world model of transitions

Right. The meta-agent is learning the **game-of-games**—a higher-order MDP where states are "which game" and transitions are "game switches." It doesn't need to predict Breakout brick physics, just that we're in a Breakout-like regime.

```
┌─────────────────────────────────────────────────────────────┐
│            META-AGENT: ABSTRACT WORLD MODEL                  │
│                                                              │
│  Level of abstraction:                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Expert WM:  "ball will hit brick at (x,y) in 3 frames"│    │
│  │ Meta WM:    "we are in a paddle-and-ball game"        │    │
│  │             "next game is likely a shooter"           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  The meta-agent compresses away game-specific dynamics       │
│  and retains only game-identity and transition structure     │
└─────────────────────────────────────────────────────────────┘
```

**Concrete architecture:**

```python
class MetaRSSM(nn.Module):
    """
    Abstract world model for game-of-games dynamics.
    Trained unsupervised on observation stream.
    Reward signal from expert performance (night update).
    """
    
    # Encoder: frames → abstract features (not pixel-reconstructible)
    encoder = CNN(
        in_channels=4,          # frame stack
        channels=[32, 64, 64],  # small, we don't need detail
        kernels=[8, 4, 3],
        strides=[4, 2, 1],
        out_dim=256
    )  # ~500K params
    
    # Recurrent backbone: integrates over time
    gru = GRU(
        input_dim=256 + 32,     # encoder out + prev stochastic
        hidden_dim=256
    )  # ~600K params
    
    # Stochastic state: VQ-VAE style discrete codes
    vq = VectorQuantize(
        codebook_size=64,       # 64 "game prototypes"
        code_dim=32,
        commitment_cost=0.25
    )  # ~2K params
    
    # Prior: p(z_t | h_t) - what game do I predict from dynamics alone?
    prior_net = MLP(256 → 64)  # logits over codes
    
    # Posterior: q(z_t | h_t, x_t) - what game given observation?
    posterior_net = MLP(256 + 256 → 64)
    
    # Transition predictor: p(g_{t+1} | h_t, z_t) for prefetch
    transition_net = MLP(256 + 32 → 64)  # dist over next game-code
    
    # Total: ~1.5M params → ~6MB
```

**The key outputs and their uses:**

| Output | Computation | Use |
|--------|-------------|-----|
| `h_t` | GRU hidden state | Temporal context for all predictions |
| `z_t` | VQ code (posterior) | Expert retrieval key |
| `KL(post \|\| prior)` | Divergence spike | Switch detection trigger |
| `p(z_{t+1} \| h_t)` | Transition net | Prefetch distribution |

**Training (unsupervised, runs continuously during day):**

```python
def meta_loss(self, trajectory):
    loss = 0
    h = self.gru.init_hidden()
    
    for x_t, x_tp1 in pairs(trajectory):
        # Encode
        e_t = self.encoder(x_t)
        
        # Posterior (uses observation)
        post_logits = self.posterior_net(cat(h, e_t))
        z_t, indices, commit_loss = self.vq(post_logits)
        
        # Prior (dynamics only)
        prior_logits = self.prior_net(h)
        
        # KL between categorical distributions
        kl = kl_divergence(post_logits, prior_logits)
        
        # Update recurrent state
        h = self.gru(cat(e_t, z_t), h)
        
        # Transition prediction (next game code)
        trans_logits = self.transition_net(cat(h, z_t))
        
        # Next step posterior as target
        e_tp1 = self.encoder(x_tp1)
        post_tp1 = self.posterior_net(cat(h, e_tp1))
        trans_loss = cross_entropy(trans_logits, post_tp1.argmax())
        
        loss += kl + commit_loss + trans_loss
    
    return loss
```

**Night update (meta-RL on expert performance):**

```python
def night_update(self, day_trajectory, cumulative_reward):
    """
    REINFORCE-style update on meta-agent's decisions.
    
    The 'actions' are:
    - When to trigger switch (threshold on KL)
    - Which expert to load (z_t embedding)
    - Prefetch decisions (transition_net predictions)
    
    Reward is cumulative expert performance over K games.
    """
    
    # Compute advantages (could use baseline)
    advantage = cumulative_reward - self.baseline
    self.baseline = 0.99 * self.baseline + 0.01 * cumulative_reward
    
    # Policy gradient on switch threshold? Or just tune?
    # For now: tune KL threshold as hyperparameter
    
    # The transition_net already gets supervised signal during day
    # Night update reinforces: "your game embeddings led to good retrieval"
    
    # Key insight: if expert performance is bad after a switch,
    # either (a) wrong expert loaded, or (b) switch too late
    # The embedding space learns to separate games that need different experts
```

**The elegant part:** The VQ codebook naturally clusters games by "what kind of expert do I need" rather than surface features. Two visually different games that require similar policies might map to the same code—which is exactly what you want for expert reuse.

Should I write up the full expert management loop (save/load/create logic with the ε-threshold)?
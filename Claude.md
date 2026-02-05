# Out-of-Core MoE Atari - Development Notes

## Architecture Overview

Hierarchical Mixture of Experts for continual learning across Atari games:
- **Meta-Agent (RSSM)**: Always in GPU memory, detects game changes, selects experts
- **Expert Library**: Game-playing experts stored on disk, swapped in/out
- **Day/Night Cycle**: Day = play K games with expert training; Night = meta-agent update

## Current Issue: RSSM Not Learning Expert Selection

**Problem**: The RSSM's `night_update()` computes an advantage but never applies a gradient. Expert selection is effectively random with respect to reward signal.

The RSSM is trained on unsupervised losses only:
1. KL divergence (prior vs posterior) - learns to predict game from dynamics
2. VQ commitment loss - learns discrete codebook
3. Transition prediction - learns game switch patterns

**Missing**: Gradient signal connecting expert selection → reward outcome

---

## Dev Plan: REINFORCE for Expert Selection

### Goal
Backpropagate reward signal through expert selection decisions so the RSSM learns which embeddings lead to good expert-game matches.

### Design

**Day Phase Changes:**
1. When RSSM produces a game embedding, compute log probability of the selected expert under the current policy
2. Store `(log_prob, expert_id, game_name)` for each expert selection during the day
3. Track per-game rewards separately

**Night Phase Changes:**
1. Compute advantage = cumulative_reward - baseline (already done)
2. Compute REINFORCE loss: `-advantage * sum(log_probs)`
3. Backprop through RSSM and update parameters

### Key Question: What is the "action" and "policy"?

Current flow:
```
obs → RSSM.encoder → posterior_net → VQ quantize → code_idx
                                                      ↓
                                          expert_manager.retrieve_or_create(embedding, code_idx)
                                                      ↓
                                          cosine similarity → select expert
```

The stochastic decision point is the VQ codebook selection. But VQ uses argmin (deterministic), so we need to either:

**Option A: Treat prior distribution as policy**
- Prior network outputs `p(code | h_t)` as softmax over codebook
- Use this as the policy for REINFORCE
- Log prob = `log prior_probs[selected_code_idx]`
- Pro: Clean separation, prior learns to predict good codes
- Con: Posterior (which actually selects) may diverge from prior

**Option B: Add stochasticity to posterior**
- Sample from posterior distribution instead of VQ argmin
- Log prob = log probability of sampled code
- Pro: Direct gradient through actual selection
- Con: Changes inference behavior, may destabilize VQ

**Option C: Gumbel-softmax (future consideration)**
- Replace VQ hard selection with Gumbel-softmax relaxation
- Allows straight-through gradients from expert performance
- Pro: End-to-end differentiable
- Con: More complex, changes codebook dynamics

**Decision: Start with Option A** (prior as policy)
- Minimal code changes
- Prior already exists and outputs code distribution
- Night update encourages prior to assign high probability to codes that led to good rewards
- Can swap to Gumbel-softmax later if needed

### Implementation Plan

#### 1. Modify MetaRSSM
- Add `get_selection_log_prob(state, code_idx)` method
- Returns `log p(code_idx | h)` from prior network

#### 2. Modify DayNightTrainer / Notebook Training Loop
- Create `day_selections: List[Tuple[log_prob, reward]]` buffer
- After each expert selection, store `log_prob = meta_agent.get_selection_log_prob(state, code_idx)`
- After each game, associate stored log_probs with game reward

#### 3. Modify night_update()
- Accept list of (log_prob, reward) tuples
- Compute per-selection advantages (reward - baseline)
- Compute REINFORCE loss: `-mean(advantage * log_prob)`
- Return loss for external optimizer step (or do internal step)

#### 4. Update Training Loop
- Call `meta_optimizer.zero_grad()` before night_update
- Call `loss.backward()` and `meta_optimizer.step()` after

### Files to Modify

1. **`core/meta_rssm.py`**
   - Add `get_selection_log_prob(self, state, code_idx) -> torch.Tensor`
   - Modify `night_update()` to compute and return REINFORCE loss

2. **`train.py`**
   - Track selection log_probs during day phase
   - Pass to night_update, apply gradient

3. **`ooc_moe_colab.ipynb`**
   - Mirror changes from train.py

### Implementation Note: Avoiding Stale Gradients

**Problem**: During day phase, we update RSSM parameters (unsupervised loss) after computing selection log_probs. By night, the gradient graph is stale.

**Solution**: Store `(h_state.detach(), code_idx)` during the day, then recompute log_probs at night using current parameters. This gives fresh gradients while preserving the actual selections made.

```python
# Day phase: store detached state and selection
selection_data.append((meta_state.h.detach().clone(), game_code.detach().clone()))

# Night phase: recompute log_probs with current params
for h_state, code_idx in selection_data:
    prior_logits = self.prior_net(h_state)  # Fresh forward pass
    log_prob = F.log_softmax(prior_logits, dim=-1)
    selected_log_prob = log_prob.gather(1, code_idx.unsqueeze(-1))
```

### Verification

1. No RuntimeError about stale gradients
2. After training, prior distribution should shift toward codes that led to high rewards
3. Expert selection should become more consistent (fewer experts for same games)
4. Performance should improve as RSSM learns better game→expert mappings

---

## Future Consideration: Gumbel-Softmax

If REINFORCE has high variance or slow convergence, consider:

```python
# In VectorQuantize.forward():
def forward(self, z, temperature=1.0, hard=True):
    # Compute logits (negative distances)
    logits = -distances  # (batch, codebook_size)

    if self.training:
        # Gumbel-softmax: differentiable sampling
        y_soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
        if hard:
            # Straight-through: hard in forward, soft in backward
            idx = y_soft.argmax(dim=-1)
            y_hard = F.one_hot(idx, self.codebook_size).float()
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        quantized = y @ self.codebook.weight
    else:
        # Inference: hard selection
        idx = logits.argmax(dim=-1)
        quantized = self.codebook(idx)

    return quantized, idx, loss
```

This would allow end-to-end gradients from expert value estimates through code selection to RSSM parameters.

---

## Other Notes

- Unified action space implemented: 6 actions for Breakout+Pong+SpaceInvaders, ~18 for full 8-game set
- Expert fragmentation was high (23 experts for 8 games) — lowered ε helped concentrate training
- Pong consistently stuck at -21 (needs investigation, likely sparse reward + long horizon issue)

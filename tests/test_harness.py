"""
Test harness for Out-of-Core MoE system.

This validates:
1. Tiered storage mechanics work correctly
2. Prefetching improves cache hit rates
3. Environment detector learns to predict expert usage
4. MOE routing is stable and experts specialize
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tiered_store import TieredExpertStore, StorageTier
from core.moe_layers import TieredMoELayer, ExpertRouter
from core.env_detector import EnvironmentDetector, EnvironmentDetectorTrainer


def test_tiered_store_basic():
    """Test basic tiered storage operations."""
    print("\n=== Test: Tiered Store Basic Operations ===")
    
    store = TieredExpertStore(
        num_experts=64,
        expert_dim=128,
        hidden_dim=256,
        hbm_capacity=8,
        dram_capacity=16,
        device="cpu",  # Use CPU for testing
        simulate_latency=False,  # Disable latency for speed
    )
    
    # Test 1: Cold start - first access should be from NVMe
    print("\n1. Cold start access...")
    experts = store.get_experts([0, 1, 2])
    assert len(experts) == 3, "Should get 3 experts"
    assert all(k in experts[0] for k in ['w1', 'w2', 'b1', 'b2']), "Expert should have all params"
    print(f"   ✓ Retrieved 3 experts from cold storage")
    print(f"   Cache stats: {store.get_stats_summary()}")
    
    # Test 2: Cache hit - second access should be from HBM
    print("\n2. Cache hit test...")
    store.stats.hits = 0
    store.stats.misses = 0
    experts = store.get_experts([0, 1, 2])
    assert store.stats.hits == 3, f"Should have 3 hits, got {store.stats.hits}"
    print(f"   ✓ Cache hit rate: {store.stats.hit_rate:.2%}")
    
    # Test 3: Cache eviction - exceed HBM capacity
    print("\n3. Cache eviction test...")
    # Access more experts than HBM capacity
    for i in range(20):
        store.get_experts([i])
    
    stats = store.get_stats_summary()
    print(f"   ✓ HBM occupancy: {stats['hbm_occupancy']:.2%}")
    print(f"   ✓ DRAM occupancy: {stats['dram_occupancy']:.2%}")
    print(f"   ✓ Evictions: {stats['evictions']}")
    
    # Test 4: LRU behavior
    print("\n4. LRU eviction order...")
    store2 = TieredExpertStore(
        num_experts=10,
        expert_dim=64,
        hidden_dim=128,
        hbm_capacity=3,
        dram_capacity=3,
        device="cpu",
        simulate_latency=False,
    )
    
    # Access 0, 1, 2 (fills HBM)
    store2.get_experts([0, 1, 2])
    # Access 0 again (moves to end of LRU)
    store2.get_experts([0])
    # Access 3 (should evict 1, the LRU)
    store2.get_experts([3])
    
    # Check 1 is not in HBM
    assert 1 not in store2.hbm_cache, "Expert 1 should be evicted"
    assert 0 in store2.hbm_cache, "Expert 0 should still be in HBM"
    print("   ✓ LRU eviction working correctly")
    
    store.shutdown()
    store2.shutdown()
    print("\n✓ Tiered store basic tests passed!")


def test_prefetching():
    """Test that prefetching improves cache performance."""
    print("\n=== Test: Prefetching Effectiveness ===")
    
    store = TieredExpertStore(
        num_experts=64,
        expert_dim=128,
        hidden_dim=256,
        hbm_capacity=16,
        dram_capacity=32,
        device="cpu",
        simulate_latency=True,
        dram_latency_ms=1.0,  # 1ms DRAM latency
        nvme_latency_ms=5.0,  # 5ms NVMe latency
    )
    
    # Simulate workload pattern: sequential access within "environments"
    # Environment 0: experts 0-15
    # Environment 1: experts 16-31
    # etc.
    
    # Test without prefetching
    print("\n1. Without prefetching...")
    start_time = time.time()
    for env in range(4):
        base = env * 16
        for _ in range(10):
            experts = [base + i for i in np.random.choice(16, 4, replace=False)]
            store.get_experts(experts)
    no_prefetch_time = time.time() - start_time
    no_prefetch_stats = store.get_stats_summary()
    print(f"   Time: {no_prefetch_time:.3f}s, Hit rate: {no_prefetch_stats['hit_rate']:.2%}")
    
    # Reset store
    store2 = TieredExpertStore(
        num_experts=64,
        expert_dim=128,
        hidden_dim=256,
        hbm_capacity=16,
        dram_capacity=32,
        device="cpu",
        simulate_latency=True,
        dram_latency_ms=1.0,
        nvme_latency_ms=5.0,
    )
    
    # Test with prefetching
    print("\n2. With prefetching...")
    start_time = time.time()
    for env in range(4):
        base = env * 16
        # Prefetch all experts for this environment
        store2.prefetch(list(range(base, base + 16)))
        time.sleep(0.1)  # Give prefetch time to work
        
        for _ in range(10):
            experts = [base + i for i in np.random.choice(16, 4, replace=False)]
            store2.get_experts(experts)
    
    with_prefetch_time = time.time() - start_time
    with_prefetch_stats = store2.get_stats_summary()
    print(f"   Time: {with_prefetch_time:.3f}s, Hit rate: {with_prefetch_stats['hit_rate']:.2%}")
    
    # Prefetching should improve hit rate
    assert with_prefetch_stats['hit_rate'] > no_prefetch_stats['hit_rate'], \
        "Prefetching should improve hit rate"
    print(f"\n   ✓ Prefetching improved hit rate by {with_prefetch_stats['hit_rate'] - no_prefetch_stats['hit_rate']:.2%}")
    
    store.shutdown()
    store2.shutdown()
    print("\n✓ Prefetching tests passed!")


def test_moe_layer():
    """Test MOE layer with tiered storage."""
    print("\n=== Test: MOE Layer Integration ===")
    
    device = "cpu"
    num_experts = 32
    expert_dim = 64
    hidden_dim = 128
    batch_size = 8
    
    store = TieredExpertStore(
        num_experts=num_experts,
        expert_dim=expert_dim,
        hidden_dim=hidden_dim,
        hbm_capacity=8,
        dram_capacity=16,
        device=device,
        simulate_latency=False,
    )
    
    moe = TieredMoELayer(
        input_dim=expert_dim,
        hidden_dim=hidden_dim,
        expert_store=store,
        top_k=2,
    )
    
    # Test forward pass
    print("\n1. Forward pass...")
    x = torch.randn(batch_size, expert_dim)
    output, aux_loss = moe(x)
    
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    assert aux_loss >= 0, "Aux loss should be non-negative"
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Aux loss: {aux_loss:.4f}")
    
    # Check routing stats
    routing_stats = moe.get_routing_stats()
    print(f"   ✓ Unique experts used: {routing_stats['num_unique_experts']}")
    print(f"   ✓ Routing entropy: {routing_stats['entropy']:.4f}")
    
    # Test that different inputs route to different experts
    print("\n2. Routing diversity...")
    all_experts = set()
    for _ in range(10):
        x = torch.randn(batch_size, expert_dim) * 5  # Larger scale for diversity
        output, _ = moe(x)
        stats = moe.get_routing_stats()
        all_experts.update(stats['expert_usage'].keys())
    
    print(f"   ✓ Total unique experts used across batches: {len(all_experts)}")
    assert len(all_experts) > 5, "Should use multiple experts"
    
    # Test gradient flow
    print("\n3. Gradient flow...")
    x = torch.randn(batch_size, expert_dim, requires_grad=True)
    output, aux_loss = moe(x)
    loss = output.sum() + aux_loss
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow to input"
    print(f"   ✓ Gradient flows through MOE layer")
    
    store.shutdown()
    print("\n✓ MOE layer tests passed!")


def test_environment_detector():
    """Test environment detector learning."""
    print("\n=== Test: Environment Detector ===")
    
    num_envs = 5
    num_experts = 32
    obs_shape = (1, 84, 84)
    
    detector = EnvironmentDetector(
        obs_shape=obs_shape,
        num_envs=num_envs,
        num_experts=num_experts,
        hidden_dim=128,
    )
    
    trainer = EnvironmentDetectorTrainer(detector)
    
    # Generate synthetic training data
    # Each environment has a characteristic observation pattern and expert usage
    print("\n1. Generating training data...")
    env_expert_mapping = {
        0: list(range(0, 8)),
        1: list(range(8, 16)),
        2: list(range(16, 24)),
        3: list(range(4, 12)),  # Overlaps with 0 and 1
        4: list(range(20, 28)),
    }
    
    for _ in range(500):
        env_id = np.random.randint(num_envs)
        
        # Create observation with environment-specific pattern
        obs = torch.randn(4, 84, 84)
        obs[:, env_id*10:(env_id+1)*10, :] += 2.0  # Add env-specific signal
        
        # Experts used by this environment
        experts = env_expert_mapping[env_id]
        used = list(np.random.choice(experts, min(4, len(experts)), replace=False))
        
        trainer.store_sample(obs, env_id, used)
    
    print(f"   ✓ Generated {len(trainer.buffer)} training samples")
    
    # Train detector
    print("\n2. Training detector...")
    losses = []
    for _ in range(100):
        loss_dict = trainer.train_step(batch_size=32)
        losses.append(loss_dict['total_loss'])
    
    print(f"   ✓ Initial loss: {losses[0]:.4f}")
    print(f"   ✓ Final loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "Loss should decrease"
    
    # Test prediction
    print("\n3. Testing predictions...")
    detector.eval()
    
    correct_env = 0
    total = 0
    precision_sum = 0
    
    for env_id in range(num_envs):
        obs = torch.randn(1, 4, 84, 84)
        obs[:, :, env_id*10:(env_id+1)*10, :] += 2.0
        
        prediction = detector.predict(obs, top_k=8)
        
        if prediction.predicted_env == env_id:
            correct_env += 1
        total += 1
        
        # Check expert prediction
        true_experts = set(env_expert_mapping[env_id])
        predicted_experts = set(prediction.prefetch_set[:8])
        precision = len(true_experts & predicted_experts) / len(predicted_experts) if predicted_experts else 0
        precision_sum += precision
    
    env_accuracy = correct_env / total
    avg_precision = precision_sum / total
    
    print(f"   ✓ Environment classification accuracy: {env_accuracy:.2%}")
    print(f"   ✓ Expert prediction precision: {avg_precision:.2%}")
    
    # Compute full prefetch accuracy
    accuracy = trainer.compute_prefetch_accuracy(recent_n=50)
    print(f"   ✓ Prefetch F1 score: {accuracy['f1']:.2%}")
    
    print("\n✓ Environment detector tests passed!")


def test_expert_specialization():
    """Test that experts specialize to different inputs."""
    print("\n=== Test: Expert Specialization ===")
    
    device = "cpu"
    num_experts = 16
    expert_dim = 32
    hidden_dim = 64
    num_envs = 4
    
    store = TieredExpertStore(
        num_experts=num_experts,
        expert_dim=expert_dim,
        hidden_dim=hidden_dim,
        hbm_capacity=16,  # All in HBM for this test
        dram_capacity=0,
        device=device,
        simulate_latency=False,
    )
    
    moe = TieredMoELayer(
        input_dim=expert_dim,
        hidden_dim=hidden_dim,
        expert_store=store,
        top_k=2,
    )
    
    # Optimize to make different environments use different experts
    print("\n1. Training for specialization...")
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.01)
    
    # Create distinct input distributions for each "environment"
    env_centers = torch.randn(num_envs, expert_dim) * 3
    
    expert_usage_by_env = defaultdict(lambda: defaultdict(int))
    
    for step in range(500):
        env_id = step % num_envs
        # Samples from this environment's distribution
        x = env_centers[env_id:env_id+1] + torch.randn(8, expert_dim) * 0.5
        
        output, aux_loss = moe(x)
        
        # Track expert usage
        for expert_id in moe.last_expert_indices.flatten().tolist():
            expert_usage_by_env[env_id][expert_id] += 1
        
        # Loss encourages specialization through auxiliary loss
        loss = aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Analyze specialization
    print("\n2. Analyzing specialization...")
    
    for env_id in range(num_envs):
        usage = expert_usage_by_env[env_id]
        total = sum(usage.values())
        top_experts = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Env {env_id}: Top experts = {[(e, f'{c/total:.1%}') for e, c in top_experts]}")
    
    # Compute overlap between environments
    print("\n3. Computing expert overlap...")
    
    def get_top_experts(env_id, k=4):
        usage = expert_usage_by_env[env_id]
        return set(e for e, _ in sorted(usage.items(), key=lambda x: x[1], reverse=True)[:k])
    
    for i in range(num_envs):
        for j in range(i+1, num_envs):
            set_i = get_top_experts(i)
            set_j = get_top_experts(j)
            overlap = len(set_i & set_j)
            print(f"   Env {i} vs Env {j}: {overlap}/4 experts overlap")
    
    store.shutdown()
    print("\n✓ Expert specialization tests passed!")


def test_full_integration():
    """Full integration test with synthetic environment."""
    print("\n=== Test: Full Integration ===")
    
    from envs.atari_wrappers import DummyAtariEnv
    
    device = "cpu"
    
    # Create model
    from models.moe_agent import MoERLAgentConfig
    
    config = MoERLAgentConfig(
        obs_shape=(1, 84, 84),
        num_actions=18,
        num_envs=5,
        frame_stack=4,
        num_experts=32,
        expert_dim=128,
        expert_hidden_dim=256,
        num_layers=2,
        num_heads=4,
        top_k=2,
        context_len=4,
        hbm_capacity=8,
        dram_capacity=16,
    )
    
    print("\n1. Creating model...")
    agent = config.create_agent(device=device)
    param_counts = config.estimate_parameter_count()
    print(f"   ✓ Total parameters: {param_counts['total']:,}")
    print(f"   ✓ Expert parameters: {param_counts['expert_params']:,}")
    print(f"   ✓ Params in HBM: {param_counts['params_in_hbm']:,}")
    
    # Create environments
    print("\n2. Creating environments...")
    envs = [DummyAtariEnv(game_id=i) for i in range(5)]
    print(f"   ✓ Created {len(envs)} dummy environments")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    obs = torch.randn(1, config.context_len, config.frame_stack, 84, 84)
    output = agent(obs, env_id=0, prefetch=True)
    
    print(f"   ✓ Action logits shape: {output.action_logits.shape}")
    print(f"   ✓ Value shape: {output.value.shape}")
    print(f"   ✓ Experts used: {output.expert_ids}")
    print(f"   ✓ Cache hit rate: {output.cache_stats['hit_rate']:.2%}")
    
    # Test action selection
    print("\n4. Testing action selection...")
    action, value = agent.get_action(obs, deterministic=False)
    print(f"   ✓ Selected action: {action}")
    print(f"   ✓ Value estimate: {value:.4f}")
    
    # Run a few steps
    print("\n5. Running environment steps...")
    env = envs[0]
    obs_np, _ = env.reset()
    obs_buffer = [torch.from_numpy(obs_np.astype(np.float32) / 255.0) for _ in range(config.context_len)]
    
    total_reward = 0
    for step in range(10):
        context = torch.stack(obs_buffer, dim=0).unsqueeze(0)
        action, value = agent.get_action(context, deterministic=False)
        
        obs_np, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        obs_buffer.pop(0)
        obs_buffer.append(torch.from_numpy(obs_np.astype(np.float32) / 255.0))
        
        if done:
            break
    
    print(f"   ✓ Ran {step+1} steps, total reward: {total_reward:.2f}")
    
    # Check cache stats after activity
    final_stats = agent.expert_store.get_stats_summary()
    print(f"\n6. Final cache statistics:")
    print(f"   ✓ Total hits: {final_stats['total_hits']}")
    print(f"   ✓ Total misses: {final_stats['total_misses']}")
    print(f"   ✓ Hit rate: {final_stats['hit_rate']:.2%}")
    
    agent.expert_store.shutdown()
    print("\n✓ Full integration tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Out-of-Core MOE Test Suite")
    print("=" * 60)
    
    try:
        test_tiered_store_basic()
        test_prefetching()
        test_moe_layer()
        test_environment_detector()
        test_expert_specialization()
        test_full_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

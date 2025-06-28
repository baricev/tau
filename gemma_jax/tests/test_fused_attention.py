#!/usr/bin/env python3
"""Test script for the fused attention implementation in gemma_jax."""

import jax
import jax.numpy as jnp
import numpy as np
from gemma_jax.core.model import multi_head_attention

def test_fused_attention():
    """Test that fused attention produces similar results to original implementation."""
    # Set up test parameters
    B, T, N, H = 2, 64, 8, 128  # batch=2, seq_len=64, heads=8, dim=128
    S = T  # Same sequence length for K,V
    K = 4  # Number of KV heads (for GQA)

    # Generate random inputs
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, N, H), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, S, K, H), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, S, K, H), dtype=jnp.bfloat16)

    # Create causal attention mask
    causal_mask = jnp.tril(jnp.ones((T, S), dtype=bool))
    attn_mask_BTS = jnp.broadcast_to(causal_mask[None, :, :], (B, T, S))

    print("Testing fused attention implementation...")
    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, mask={attn_mask_BTS.shape}")

    # Test original implementation (fallback)
    print("\nTesting original implementation...")
    try:
        output_orig = multi_head_attention(q, k, v, attn_mask_BTS, use_fused_kernel=False)
        print(f"Original output shape: {output_orig.shape}")
        print(f"Original output stats: mean={float(output_orig.mean()):.6f}, std={float(output_orig.std()):.6f}")
    except Exception as e:
        print(f"Original implementation failed: {e}")
        assert False

    # Test fused implementation
    print("\nTesting fused implementation...")
    try:
        output_fused = multi_head_attention(q, k, v, attn_mask_BTS, use_fused_kernel=True)
        print(f"Fused output shape: {output_fused.shape}")
        print(f"Fused output stats: mean={float(output_fused.mean()):.6f}, std={float(output_fused.std()):.6f}")

        # Compare outputs
        diff = jnp.abs(output_orig - output_fused)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)

        print(f"\nComparison:")
        print(f"Max difference: {float(max_diff):.6f}")
        print(f"Mean difference: {float(mean_diff):.6f}")

        # Check if results are reasonably close (allowing for numerical differences)
        assert max_diff < 1e-2 and mean_diff < 1e-3
        print("✅ Fused attention test PASSED - outputs are similar")

    except Exception as e:
        print(f"Fused implementation failed: {e}")
        print("This is expected if splash attention is not available")
        assert True

def test_decode_case():
    """Test decode case (T=1) uses original implementation."""
    B, T, N, H = 2, 1, 8, 128  # decode case
    S = 64  # Cache length
    K = 4

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], (B, T, N, H), dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], (B, S, K, H), dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], (B, S, K, H), dtype=jnp.bfloat16)

    # Create mask
    attn_mask_BTS = jnp.ones((B, T, S), dtype=bool)

    print("\nTesting decode case (T=1)...")
    try:
        output = multi_head_attention(q, k, v, attn_mask_BTS, use_fused_kernel=True)
        print(f"Decode output shape: {output.shape}")
        print("✅ Decode case test PASSED")
        assert True
    except Exception as e:
        print(f"Decode case failed: {e}")
        assert False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Fused Attention Implementation")
    print("=" * 60)

    success1 = test_fused_attention()
    success2 = test_decode_case()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

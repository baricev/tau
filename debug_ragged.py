#!/usr/bin/env python3
"""Debug ragged attention implementation."""

import jax
import jax.numpy as jnp
import numpy as np
from gemma_jax.core.model import multi_head_attention

def debug_mask_creation():
    """Debug mask creation and conversion."""
    # Simple test case
    B, T, N, H = 2, 1, 4, 32
    S = 64
    K = 2
    
    # Test with specific lengths
    lengths = jnp.array([32, 48], dtype=jnp.int32)
    
    print("=== Debug Mask Creation ===")
    print(f"Lengths: {lengths}")
    print(f"Cache size (S): {S}")
    
    # Method 1: How I create mask in ragged attention
    pos_mask_ragged = jnp.arange(S)[None, None, :] < lengths[:, None, None]
    pos_mask_ragged = jnp.broadcast_to(pos_mask_ragged, (B, T, S))
    print(f"Ragged mask shape: {pos_mask_ragged.shape}")
    print(f"Ragged mask sum per batch: {jnp.sum(pos_mask_ragged, axis=(1,2))}")
    
    # Method 2: How mask is created in test
    pos_mask_test = jnp.arange(S)[None, None, :] < lengths[:, None, None]
    attn_mask_BTS = jnp.broadcast_to(pos_mask_test, (B, T, S))
    print(f"Test mask shape: {attn_mask_BTS.shape}")
    print(f"Test mask sum per batch: {jnp.sum(attn_mask_BTS, axis=(1,2))}")
    
    # Check if they're identical
    print(f"Masks identical: {jnp.allclose(pos_mask_ragged, attn_mask_BTS)}")
    
    # Check the lengths conversion
    lengths_recovered = jnp.sum(attn_mask_BTS[:, 0, :].astype(jnp.int32), axis=-1)
    print(f"Recovered lengths: {lengths_recovered}")
    print(f"Lengths match: {jnp.allclose(lengths, lengths_recovered)}")

def debug_attention_computation():
    """Debug the attention computation step by step."""
    print("\n=== Debug Attention Computation ===")
    
    # Simple deterministic test
    B, T, N, H = 1, 1, 2, 4
    S = 8
    K = 1
    
    # Create simple test data
    q = jnp.ones((B, T, N, H)) * 0.1  # Small values
    k = jnp.ones((B, S, K, H)) * 0.1
    v = jnp.arange(B * S * K * H).reshape(B, S, K, H).astype(jnp.float32) * 0.01
    
    length = jnp.array([4], dtype=jnp.int32)  # Only attend to first 4 positions
    
    print(f"Query shape: {q.shape}, values: {q[0,0,0,:]}")
    print(f"Key shape: {k.shape}, values: {k[0,:,0,0]}")
    print(f"Value shape: {v.shape}")
    print(f"Length: {length}")
    
    # Test masked attention
    pos_mask = jnp.arange(S)[None, None, :] < length[:, None, None]
    attn_mask_BTS = jnp.broadcast_to(pos_mask, (B, T, S))
    
    print(f"\nMask: {attn_mask_BTS[0,0,:]}")
    
    output_masked = multi_head_attention(
        q, k, v, attn_mask_BTS,
        use_fused_kernel=False,
        use_ragged_attention=False
    )
    
    output_ragged = multi_head_attention(
        q, k, v, length,
        use_fused_kernel=False,
        use_ragged_attention=True
    )
    
    print(f"\nMasked output: {output_masked[0,0,0,:]}")
    print(f"Ragged output: {output_ragged[0,0,0,:]}")
    print(f"Difference: {output_masked[0,0,0,:] - output_ragged[0,0,0,:]}")
    print(f"Max diff: {jnp.max(jnp.abs(output_masked - output_ragged))}")

if __name__ == "__main__":
    debug_mask_creation()
    debug_attention_computation()
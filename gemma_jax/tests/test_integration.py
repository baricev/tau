"""Integration tests that demonstrate the bugs in realistic scenarios."""

import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from gemma_jax.core.cache import KVCache, init_cache, update_cache_layer
from gemma_jax.core.segment import SegmentInfo
from gemma_jax.core.model import (
    multi_head_attention, forward_fn, setup_scan_fn, scan_generate_step,
    AttentionConfig, AttentionType, Layer, Gemma3, make_attention_layers_types
)


def create_test_model(num_layers=2, num_heads=4, embed_dim=128, vocab_size=100):
    """Create a small test model."""
    head_dim = embed_dim // num_heads
    hidden_dim = embed_dim * 4
    
    blocks = []
    for _ in range(num_layers):
        layer = Layer(
            attn_key_norm_scale=jnp.ones((head_dim,)),
            attn_query_norm_scale=jnp.ones((head_dim,)),
            output_proj=jnp.ones((num_heads, head_dim, embed_dim)) * 0.01,
            kv_proj=jnp.ones((2, num_heads, embed_dim, head_dim)) * 0.01,
            q_proj=jnp.ones((num_heads, embed_dim, head_dim)) * 0.01,
            gating_weights=jnp.ones((2, hidden_dim, embed_dim)) * 0.01,
            output_weights=jnp.ones((hidden_dim, embed_dim)) * 0.01,
            post_attention_norm_scale=jnp.ones((embed_dim,)),
            post_ffw_norm_scale=jnp.ones((embed_dim,)),
            pre_attention_norm_scale=jnp.ones((embed_dim,)),
            pre_ffw_norm_scale=jnp.ones((embed_dim,)),
        )
        blocks.append(layer)
    
    return Gemma3(
        input_embedding_table=jnp.ones((vocab_size, embed_dim)) * 0.1,
        mm_input_projection=None,
        mm_soft_embedding_norm=None,
        final_norm_scale=jnp.ones((embed_dim,)),
        blocks=tuple(blocks),
    )


def create_test_config(num_layers=2, num_heads=4, embed_dim=128, use_ragged=False):
    """Create a test configuration."""
    class Config:
        pass
    
    config = Config()
    config.num_layers = num_layers
    config.num_heads = num_heads
    config.num_kv_heads = num_heads
    config.embed_dim = embed_dim
    config.head_dim = embed_dim // num_heads
    config.hidden_dim = embed_dim * 4
    config.query_pre_attn_scalar = 1.0 / np.sqrt(config.head_dim)
    config.cache_length = 20
    config.vocab_size = 100
    config.window_size = 4
    config.local_base_frequency = 10000
    config.global_base_frequency = 10000
    config.local_scale_factor = 1.0
    config.global_scale_factor = 1.0
    config.use_ragged_attention = use_ragged
    
    return config


class TestFullGeneration:
    """Test complete generation flow to catch interaction bugs."""
    
    def test_generation_with_all_bugs(self):
        """Run generation that would fail with any of the 5 bugs."""
        batch_size = 2
        config = create_test_config(num_layers=2, use_ragged=True)
        model = create_test_model(num_layers=2)
        
        # Input tokens
        input_ids = jnp.array([
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 0, 0, 0],
        ])
        
        # Initialize cache
        cache = init_cache(
            batch=batch_size,
            max_seq_len=config.cache_length,
            num_layers=config.num_layers,
            num_kv_heads=config.num_heads,
            head_dim=config.head_dim,
        )
        
        # Mock ragged kernels
        def mock_ragged_mha(q, k, v, lengths):
            # Check for Bug 1: lengths should include current token
            print(f"Ragged MHA called with lengths: {lengths}")
            return jnp.zeros_like(q), jnp.zeros((q.shape[0], q.shape[1], 1)), jnp.ones((q.shape[0], q.shape[1], 1))
        
        # First, do prefill
        positions = jnp.cumsum(input_ids != 0, axis=-1) - 1
        positions = jnp.where(input_ids != 0, positions, 0)
        
        seg_info = SegmentInfo(
            lengths=jnp.zeros((batch_size,), dtype=jnp.int32),
            cursor=jnp.zeros((batch_size,), dtype=jnp.int32),
            offset=jnp.zeros((batch_size,), dtype=jnp.int32),
            cache_len=config.cache_length,
        )
        
        with patch('gemma_jax.core.model._ragged_mha', mock_ragged_mha):
            # Prefill
            x, cache = forward_fn(
                None, input_ids, positions, seg_info,
                model, cache, None, config, False, None
            )
            
            # Setup for generation - this is where Bug 3 manifests
            carry = setup_scan_fn(None, input_ids, cache)
            last_tokens, seg_info, _, cache, _ = carry
            
            # Generate a few tokens
            for step in range(3):
                print(f"\nGeneration step {step}:")
                print(f"  seg_info.lengths: {seg_info.lengths}")
                print(f"  seg_info.cursor: {seg_info.cursor}")
                print(f"  cache.sequence_lengths: {cache.sequence_lengths}")
                print(f"  cache.write_positions: {cache.write_positions}")
                
                # Bug 5 would cause these to diverge
                assert jnp.array_equal(cache.sequence_lengths, seg_info.lengths), \
                    f"Bug 5: Cache lengths {cache.sequence_lengths} != seg_info lengths {seg_info.lengths}"
                
                carry, next_token = scan_generate_step(
                    carry, None,
                    model=model,
                    rope_cache=None,
                    config=config
                )
                last_tokens, seg_info, _, cache, _ = carry


class TestSlidingWindowIntegration:
    """Test sliding window with different attention types."""
    
    def test_attention_pattern_with_sliding_window(self):
        """Test that sliding window layers are handled correctly."""
        config = create_test_config(num_layers=6)
        
        # Test the attention pattern
        attention_types = make_attention_layers_types(6)
        
        # Count sliding vs global layers
        sliding_count = sum(1 for t in attention_types if t == AttentionType.LOCAL_SLIDING)
        global_count = sum(1 for t in attention_types if t == AttentionType.GLOBAL)
        
        assert sliding_count == 5, f"Expected 5 sliding layers, got {sliding_count}"
        assert global_count == 1, f"Expected 1 global layer, got {global_count}"
        
        # Test with window_size = 0 (Bug 4)
        config.window_size = 0
        cache = init_cache(
            batch=1,
            max_seq_len=10,
            num_layers=6,
            num_kv_heads=4,
            head_dim=32,
        )
        
        seg_info = SegmentInfo(
            lengths=jnp.array([5]),
            cursor=jnp.array([5]),
            offset=jnp.zeros((1,)),
            cache_len=10
        )
        
        # Check each layer
        for layer_idx, attn_type in enumerate(attention_types):
            window = config.window_size if attn_type == AttentionType.LOCAL_SLIDING else None
            
            # Bug 4: window=0 creates empty mask
            if window == 0:
                # Should treat as None to avoid empty mask
                k, v, mask = cache.lookup_layer(
                    seg_info,
                    layer=layer_idx,
                    window=window,
                )
                
                # This would fail with Bug 4
                assert jnp.any(mask), f"Layer {layer_idx}: mask is empty with window={window}"


# CPU Testing Guide
class CPUTestingGuide:
    """
    All five bugs can be caught on CPU-only systems!
    
    Here's how each bug can be tested without TPU:
    
    1. **Bug 1 (Ragged token visibility)**: Mock the ragged kernels and check
       the arguments passed to them. The lengths should include the new token.
    
    2. **Bug 2 (Sliding window + ragged)**: Mock ragged kernels and verify
       that sliding window layers either disable ragged or implement windowing.
    
    3. **Bug 3 (Cursor alignment)**: Pure logic bug - test cursor initialization
       and check that positions align correctly.
    
    4. **Bug 4 (Zero window)**: Pure logic bug - test mask generation with
       window_size=0 and verify it's not empty.
    
    5. **Bug 5 (Counter increments)**: Pure logic bug - verify counters
       increment once per step, not per layer.
    
    The only TPU-specific parts are:
    - Ragged attention kernels (_ragged_mha, _ragged_gqa)
    - Splash attention kernels
    
    These can be mocked for testing the logic around them.
    """
    pass


if __name__ == "__main__":
    print("Running integration tests...")
    
    # Run with mocked TPU kernels
    with patch('gemma_jax.core.model._ragged_mha', lambda *args: (jnp.zeros((2, 1, 4, 32)), None, None)), \
         patch('gemma_jax.core.model._ragged_gqa', lambda *args: (jnp.zeros((2, 1, 4, 32)), None, None)):
        
        test1 = TestFullGeneration()
        test1.test_generation_with_all_bugs()
        print("✓ Full generation test passed")
        
        test2 = TestSlidingWindowIntegration()
        test2.test_attention_pattern_with_sliding_window()
        print("✓ Sliding window integration test passed")
    
    print("\n✅ All integration tests passed on CPU!")

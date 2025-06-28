# tests/test_ring_buffer_wrap.py
"""
Ensure that when more tokens than `cache_len` are written
  • the cursor wraps,
  • the offset advances by the overflow amount, and
  • the oldest slot is overwritten with the new token.

This catches fence‑post errors that only show up after very long sequences.
"""

import jax
import jax.numpy as jnp

from gemma_jax.core.cache   import init_cache, update_cache_layer
from gemma_jax.core.segment import SegmentInfo


def test_ring_buffer_wrap_once():
    BATCH       = 1
    CACHE_LEN   = 4          # intentionally tiny
    NUM_LAYERS  = 1
    NUM_KV_HEAD = 1
    HEAD_DIM    = 1

    # 1) fresh, zero‑filled cache + metadata
    cache = init_cache(
        batch=BATCH,
        max_seq_len=CACHE_LEN,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEAD,
        head_dim=HEAD_DIM,
        dtype=jnp.bfloat16,
    )
    seg = SegmentInfo(
        lengths=jnp.zeros((BATCH,), dtype=jnp.int32),
        cursor=jnp.zeros((BATCH,), dtype=jnp.int32),
        offset=jnp.zeros((BATCH,), dtype=jnp.int32),
        cache_len=CACHE_LEN,
    )

    # 2) write CACHE_LEN + 1 single‑token steps (forces exactly one wrap)
    for step in range(CACHE_LEN + 1):
        value = jnp.asarray(step + 1, dtype=jnp.int32)          # 1,2,3,4,5
        key_proj   = value.reshape(BATCH, 1, 1, 1)              # (B,1,K,H)
        value_proj = key_proj

        _, _, cache = update_cache_layer(
            cache,
            key_proj,
            value_proj,
            seg_info=seg,
            chunk_lens_B=jnp.ones((BATCH,), dtype=jnp.int32),
            layer=0,
            ragged=True,   # single‑token “generation” path
        )
        seg = seg.advance(1)

    # 3) assertions – CI will go red on any off‑by‑one
    ring_contents = cache.key[0, 0, :, 0, 0].astype(jnp.bfloat16)
    ring_contents = ring_contents * cache.key_scale[0, 0, :, 0]

    expected = jnp.asarray([5, 2, 3, 4], dtype=jnp.bfloat16)
    assert jnp.allclose(ring_contents, expected, atol=2e-2), (
        f"{ring_contents=} {expected=}"
    )

    # sequence length is capped at CACHE_LEN after wrap
    assert int(seg.lengths[0]) == CACHE_LEN
    # cursor wrapped to position 1 (just wrote slot 0)
    assert int(seg.cursor[0]) == 1
    # offset advanced by exactly one slot
    assert int(seg.offset[0]) == 1

    print(f"test_ring_buffer_wrap_once passed: {ring_contents=}, {seg.lengths=}, {seg.cursor=}, {seg.offset=}")


def test_update_cache_layer_counters_once():
    """Sequence length should not advance for each layer update."""
    BATCH = 1
    CACHE_LEN = 8

    cache = init_cache(
        batch=BATCH,
        max_seq_len=CACHE_LEN,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=jnp.bfloat16,
    )

    seg = SegmentInfo(
        lengths=jnp.zeros((BATCH,), dtype=jnp.int32),
        cursor=jnp.zeros((BATCH,), dtype=jnp.int32),
        offset=jnp.zeros((BATCH,), dtype=jnp.int32),
        cache_len=CACHE_LEN,
    )

    k_proj = jnp.ones((BATCH, 1, 1, 1), dtype=jnp.bfloat16)
    v_proj = jnp.ones((BATCH, 1, 1, 1), dtype=jnp.bfloat16)
    step_len = jnp.ones((BATCH,), dtype=jnp.int32)

    _, _, cache = update_cache_layer(
        cache, k_proj, v_proj,
        seg_info=seg,
        chunk_lens_B=step_len,
        layer=0,
        ragged=True,
    )
    _, _, cache = update_cache_layer(
        cache, k_proj, v_proj,
        seg_info=seg,
        chunk_lens_B=step_len,
        layer=0,
        ragged=True,
    )

    assert int(cache.sequence_lengths[0]) == 1
    assert int(cache.write_positions[0]) == 1


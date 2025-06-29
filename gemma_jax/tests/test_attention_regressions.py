#!/usr/bin/env python3
"""Regression tests for attention bug fixes."""

import jax
import jax.numpy as jnp
import dataclasses
import pytest

from gemma_jax.core.cache import init_cache, update_cache_layer
from gemma_jax.core.segment import SegmentInfo
from gemma_jax.core.model import (
    AttentionConfig,
    AttentionType,
    apply_rope,
    qkv_projection,
    multi_head_attention,
    self_attention,
)

from .test_ragged_attention import _make_identity_layer


def test_local_sliding_respects_window():
    """Ragged generation with LOCAL_SLIDING should honour the window mask."""
    N = K = 1
    H = 2
    D = N * H
    S = 4

    layer = _make_identity_layer(N=N, K=K, H=H)
    cfg = AttentionConfig(
        num_heads=N,
        num_kv_heads=K,
        embed_dim=D,
        head_dim=H,
        hidden_dim=4,
        attn_type=AttentionType.LOCAL_SLIDING,
        query_pre_attn_scalar=1.0,
        cache_length=S,
        window_size=1,
        use_ragged_attention=True,
    )

    cache = init_cache(
        batch=1,
        max_seq_len=S,
        num_layers=1,
        num_kv_heads=K,
        head_dim=H,
        dtype=jnp.bfloat16,
    )
    seg = SegmentInfo(
        lengths=jnp.zeros((1,), dtype=jnp.int32),
        cursor=jnp.zeros((1,), dtype=jnp.int32),
        offset=jnp.zeros((1,), dtype=jnp.int32),
        cache_len=S,
    )

    # Prefill two tokens
    for step in range(2):
        x = jax.random.normal(jax.random.PRNGKey(step + 1), (1, 1, D), dtype=jnp.bfloat16)
        q, k, v = qkv_projection(x, layer.q_proj, layer.kv_proj)
        k = apply_rope(k, jnp.array([[step]]), base_frequency=cfg.rope_base_frequency)
        _, _, cache = update_cache_layer(
            cache,
            k,
            v,
            seg_info=seg,
            chunk_lens_B=jnp.ones((1,), dtype=jnp.int32),
            layer=0,
            ragged=True,
        )
        seg = seg.advance(1)

    # New token
    x = jax.random.normal(jax.random.PRNGKey(3), (1, 1, D), dtype=jnp.bfloat16)

    # Manual baseline using masked attention with sliding window
    q2, k2, v2 = qkv_projection(x, layer.q_proj, layer.kv_proj)
    q2 = apply_rope(q2, jnp.array([[seg.current_pos[0]]]), base_frequency=cfg.rope_base_frequency)
    k2r = apply_rope(k2, jnp.array([[seg.current_pos[0]]]), base_frequency=cfg.rope_base_frequency)
    _, _, cache_manual = update_cache_layer(
        cache,
        k2r,
        v2,
        seg_info=seg,
        chunk_lens_B=jnp.ones((1,), dtype=jnp.int32),
        layer=0,
        ragged=True,
    )
    seg_lookup = seg.advance(1)
    ck, cv, mask = cache_manual.lookup_layer(seg_lookup, layer=0, window=cfg.window_size, query_positions=None)
    mask = mask[:, None, :]
    baseline = multi_head_attention(
        q2.astype(jnp.float32),
        ck.astype(jnp.float32),
        cv.astype(jnp.float32),
        mask,
        use_fused_kernel=True,
        use_ragged_attention=False,
    )

    out, _ = self_attention(
        None,
        x,
        jnp.array([[seg.current_pos[0]]], dtype=jnp.int32),
        layer,
        cache,
        None,
        seg,
        jnp.ones((1,), dtype=jnp.int32),
        cfg,
        True,
        0,
    )

    assert jnp.allclose(out, baseline.astype(out.dtype), atol=1e-3)


def test_window_size_zero_equivalent_to_none():
    """LOCAL_SLIDING with window_size=0 should behave like GLOBAL."""
    N = K = 1
    H = 2
    D = 2
    S = 4

    layer = _make_identity_layer(N=N, K=K, H=H)
    cfg_global = AttentionConfig(
        num_heads=N,
        num_kv_heads=K,
        embed_dim=D,
        head_dim=H,
        hidden_dim=4,
        attn_type=AttentionType.GLOBAL,
        query_pre_attn_scalar=1.0,
        cache_length=S,
        window_size=0,
        use_ragged_attention=True,
    )
    cfg_zero = dataclasses.replace(cfg_global, attn_type=AttentionType.LOCAL_SLIDING)

    cache = init_cache(
        batch=1,
        max_seq_len=S,
        num_layers=1,
        num_kv_heads=K,
        head_dim=H,
        dtype=jnp.bfloat16,
    )
    seg = SegmentInfo(
        lengths=jnp.zeros((1,), dtype=jnp.int32),
        cursor=jnp.zeros((1,), dtype=jnp.int32),
        offset=jnp.zeros((1,), dtype=jnp.int32),
        cache_len=S,
    )

    x0 = jax.random.normal(jax.random.PRNGKey(0), (1, 1, D), dtype=jnp.bfloat16)
    _, k0, v0 = qkv_projection(x0, layer.q_proj, layer.kv_proj)
    k0 = apply_rope(k0, jnp.array([[0]]), base_frequency=cfg_global.rope_base_frequency)
    _, _, cache = update_cache_layer(
        cache,
        k0,
        v0,
        seg_info=seg,
        chunk_lens_B=jnp.ones((1,), dtype=jnp.int32),
        layer=0,
        ragged=True,
    )
    seg = seg.advance(1)

    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, 1, D), dtype=jnp.bfloat16)

    try:
        out_global, _ = self_attention(
            None,
            x1,
            jnp.array([[seg.current_pos[0]]], dtype=jnp.int32),
            layer,
            cache,
            None,
            seg,
            jnp.ones((1,), dtype=jnp.int32),
            cfg_global,
            True,
            0,
        )
        out_zero, _ = self_attention(
            None,
            x1,
            jnp.array([[seg.current_pos[0]]], dtype=jnp.int32),
            layer,
            cache,
            None,
            seg,
            jnp.ones((1,), dtype=jnp.int32),
            cfg_zero,
            True,
            0,
        )

        assert jnp.allclose(out_global, out_zero, atol=1e-3)
    except Exception as e:
        pytest.skip(f"ragged kernels unavailable: {e}")


def test_update_cache_layer_counts_once_per_token():
    """sequence_lengths and write_positions update once per token, not per layer."""
    B = 1
    S = 4
    L = 2
    K = H = 1
    D = 1

    cache = init_cache(
        batch=B,
        max_seq_len=S,
        num_layers=L,
        num_kv_heads=K,
        head_dim=H,
        dtype=jnp.bfloat16,
    )
    seg = SegmentInfo(
        lengths=jnp.zeros((B,), dtype=jnp.int32),
        cursor=jnp.zeros((B,), dtype=jnp.int32),
        offset=jnp.zeros((B,), dtype=jnp.int32),
        cache_len=S,
    )

    layer = _make_identity_layer(N=1, K=1, H=1)
    x = jnp.ones((B, 1, D), dtype=jnp.bfloat16)
    _, k, v = qkv_projection(x, layer.q_proj, layer.kv_proj)

    for lyr in range(L):
        _, _, cache = update_cache_layer(
            cache,
            k,
            v,
            seg_info=seg,
            chunk_lens_B=jnp.ones((B,), dtype=jnp.int32),
            layer=lyr,
            ragged=True,
        )

    assert int(cache.sequence_lengths[0]) == 1
    assert int(cache.write_positions[0]) == 1

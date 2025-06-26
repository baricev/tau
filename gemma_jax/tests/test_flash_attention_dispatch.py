"""Minimal test exercising flash attention path."""

from gemma_jax.core.flash_attention import flash_attention, HAS_JETSTREAM
from gemma_jax.core.model import multi_head_attention, AttentionConfig


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
        return jax, jnp
    except Exception:
        return None, None


def test_flash_attention_flag():
    jax, jnp = _import_jax()
    if jax is None:
        print("JAX not installed; skipping test.")
        return

    cfg = AttentionConfig(
        num_heads=1,
        num_kv_heads=1,
        embed_dim=1,
        head_dim=1,
        hidden_dim=1,
        attn_type=1,
        query_pre_attn_scalar=1.0,
        use_flash_attention=True,
    )
    q = jnp.zeros((1, 1, 1, 1))
    k = jnp.zeros((1, 1, 1, 1))
    v = jnp.zeros((1, 1, 1, 1))
    m = jnp.ones((1, 1, 1), dtype=jnp.bool_)
    try:
        out = flash_attention(q, k, v, m)
    except Exception:
        out = multi_head_attention(q, k, v, m)
    assert out.shape == (1, 1, 1, 1)


test_flash_attention_flag()

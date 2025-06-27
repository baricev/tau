import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as tpu_flash_attention,
    mha_reference,
    BlockSizes,
    DEFAULT_MASK_VALUE,
)

__all__ = ["multi_head_flash_attention"]


def multi_head_flash_attention(q: jax.Array, k: jax.Array, v: jax.Array, attn_mask: jax.Array | None) -> jax.Array:
    """Flash attention wrapper matching ``multi_head_attention`` interface.

    Args:
        q: (B, T, N, H) query tensor.
        k: (B, S, K, H) key tensor.
        v: (B, S, K, H) value tensor.
        attn_mask: (B, T, S) boolean mask where ``True`` means keep.
    Returns:
        Attention output of shape (B, T, N, H).
    """
    B, T, N, H = q.shape
    _, S, K, _ = k.shape
    G = N // K

    q_bnh = jnp.swapaxes(q, 1, 2)  # (B, N, T, H)
    k_bkh = jnp.swapaxes(k, 1, 2)  # (B, K, S, H)
    v_bkh = jnp.swapaxes(v, 1, 2)

    # repeat kv heads to match query heads
    if G > 1:
        k_bkh = jnp.repeat(k_bkh, G, axis=1)
        v_bkh = jnp.repeat(v_bkh, G, axis=1)

    ab = None
    if attn_mask is not None:
        bias = jnp.where(attn_mask[:, None, :, :], 0.0, DEFAULT_MASK_VALUE)
        ab = bias.astype(jnp.float32)

    if jax.default_backend() == "tpu":
        block_sizes = BlockSizes(block_q=T, block_k_major=S, block_k=S, block_b=1)
        out = tpu_flash_attention(
            q_bnh.astype(jnp.bfloat16),
            k_bkh.astype(jnp.bfloat16),
            v_bkh.astype(jnp.bfloat16),
            ab=ab,
            causal=False,
            block_sizes=block_sizes,
        )
    else:
        out = mha_reference(
            q_bnh.astype(jnp.bfloat16),
            k_bkh.astype(jnp.bfloat16),
            v_bkh.astype(jnp.bfloat16),
            ab=ab,
            causal=False,
        )

    return jnp.swapaxes(out, 1, 2)


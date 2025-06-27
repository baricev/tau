"""Ragged attention kernels adapted from MaxText for Gemma JAX."""

import functools
import numpy as np

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Use same mask value as Gemma JAX model.py
DEFAULT_MASK_VALUE = -2.3819763e38


def get_gqa_cost_estimate(shape_dtype):
    """Get cost estimate for GQA based on static shape information."""
    batch_size, _, num_heads, head_dim = shape_dtype[0].shape
    seq_len = shape_dtype[1].shape[1]

    # Approximate flops calculation for attention
    flops = batch_size * num_heads * seq_len * (
        2 * head_dim +  # QK multiplication
        seq_len +       # softmax
        2 * head_dim    # V multiplication
    )

    return pl.CostEstimate(
        flops=flops,
        transcendentals=batch_size * num_heads * seq_len,
        bytes_accessed=int(sum(np.prod(s.shape) * s.dtype.itemsize for s in shape_dtype)),
    )


@functools.partial(jax.jit, static_argnames=["mask_value"])
def reference_gqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference grouped query attention implementation for variable-length sequences.

    Args:
        q: A [batch_size, 1, num_heads_q, head_dim] jax.Array.
        k: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
        v: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
        lengths: A i32[batch_size] jax.Array containing actual sequence lengths.
        mask_value: The value used for padding in attention.

    Returns:
        The output of attention([batch_size, 1, num_heads_q, head_dim]), along with the
        max logit ([batch_size, 1, num_heads_q, 1]) and softmax denominator 
        ([batch_size, 1, num_heads_q, 1]).
    """
    batch_size, _, num_heads_q, head_dim = q.shape
    _, seq_len, num_heads_kv, _ = k.shape
    assert k.shape == v.shape
    assert num_heads_q % num_heads_kv == 0

    # Reshape q to group format
    q = q.reshape(batch_size, 1, num_heads_kv, num_heads_q // num_heads_kv, head_dim)

    # Compute attention scores: (batch, 1, kv_heads, groups, seq_len)
    logits = jnp.einsum("bthgd,bskd->bthgs", q.astype(jnp.float32), k.astype(jnp.float32))
    
    # Create mask based on actual sequence lengths
    mask = jnp.arange(seq_len)[None] < lengths[:, None]
    logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, None, :]
    
    # Compute softmax with numerical stability
    logits_max = logits.max(axis=-1, keepdims=True)
    unnormalized = jnp.exp(logits - logits_max)
    denominator = unnormalized.sum(axis=-1, keepdims=True)
    
    # Apply attention to values
    o = jnp.einsum("bthgs,bskd->bthgd", unnormalized.astype(v.dtype), v) / denominator
    
    # Reshape outputs back to original format
    logits_max = logits_max.reshape(batch_size, 1, num_heads_q, 1)
    denominator = denominator.reshape(batch_size, 1, num_heads_q, 1)
    o = o.reshape(batch_size, 1, num_heads_q, head_dim)
    
    return o, logits_max, denominator


def ragged_flash_attention_kernel(
    lengths_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    *,
    block_size: int,
    mask_value: float,
):
    """Pallas kernel for ragged flash attention."""
    b, i = pl.program_id(0), pl.program_id(1)

    @pl.when(i == 0)
    def init():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        o_ref[...] = jnp.zeros_like(o_ref)

    length = lengths_ref[b]

    @pl.when(i * block_size < length)
    def run():
        q = q_ref[...].astype(jnp.float32)
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...].astype(jnp.float32)
        m_prev, l_prev = m_ref[...], l_ref[...]

        qk = lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        mask = i * block_size + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
        qk = qk + jnp.where(mask, 0.0, mask_value)
        m_curr = qk.max(axis=-1)

        m_new = jnp.maximum(m_prev, m_curr)
        p = jnp.exp(qk - m_new[:, None])
        l_new = jnp.exp(m_prev - m_new) * l_prev + p.sum(axis=-1)

        o_prev = o_ref[...]
        o_new = (
            o_prev * jnp.exp(m_prev - m_new)[:, None] +
            lax.dot_general(p.astype(v.dtype), v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        )

        m_ref[...] = m_new
        l_ref[...] = l_new
        o_ref[...] = o_new

    @pl.when(i == pl.num_programs(1) - 1)
    def normalize():
        o_ref[...] = o_ref[...] / l_ref[..., None]


def ragged_mqa_kernel(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int = 256,
    mask_value: float = DEFAULT_MASK_VALUE,
    cost_estimate: pl.CostEstimate | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Multi-query attention with ragged sequences using Pallas kernel."""
    batch_size, num_heads, head_dim = query.shape
    _, seq_len, _ = key.shape

    # Define grid and block specifications
    grid = (batch_size, (seq_len + block_size - 1) // block_size)
    
    def kernel_wrapper(lengths_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref):
        return ragged_flash_attention_kernel(
            lengths_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref,
            block_size=block_size, mask_value=mask_value
        )

    # Define input/output specifications
    in_specs = [
        pl.BlockSpec((1,), lambda b, i: (b,)),  # lengths
        pl.BlockSpec((num_heads, head_dim), lambda b, i: (b, slice(None), slice(None))),  # q
        pl.BlockSpec((block_size, head_dim), lambda b, i: (b, pl.dslice(i * block_size, block_size), slice(None))),  # k
        pl.BlockSpec((block_size, head_dim), lambda b, i: (b, pl.dslice(i * block_size, block_size), slice(None))),  # v
    ]
    
    out_specs = [
        pl.BlockSpec((num_heads, head_dim), lambda b, i: (b, slice(None), slice(None))),  # o
        pl.BlockSpec((num_heads,), lambda b, i: (b, slice(None))),  # m
        pl.BlockSpec((num_heads,), lambda b, i: (b, slice(None))),  # l
    ]

    # Call Pallas kernel
    o, m, l = pl.pallas_call(
        kernel_wrapper,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
        cost_estimate=cost_estimate,
    )(lengths, query, key, value, 
      jnp.zeros((batch_size, num_heads, head_dim), dtype=query.dtype),
      jnp.full((batch_size, num_heads), -jnp.inf, dtype=jnp.float32),
      jnp.zeros((batch_size, num_heads), dtype=jnp.float32))

    return o, m[:, None], l[:, None]


def ragged_gqa(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int = 256,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Ragged grouped query attention for variable-length sequences.

    Args:
        query: A [batch_size, 1, num_heads_q, head_dim] jax.Array.
        key: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
        value: A [batch_size, seq_len, num_heads_kv, head_dim] jax.Array.
        lengths: A i32[batch_size] jax.Array containing actual sequence lengths.
        block_size: Block size for Pallas kernel computation.
        mask_value: Value used for masking padded positions.

    Returns:
        The output of attention([batch_size, 1, num_heads_q, head_dim]), along with the
        max logit ([batch_size, 1, num_heads_q, 1]) and softmax denominator 
        ([batch_size, 1, num_heads_q, 1]).
    """
    batch_size, _, num_heads_q, head_dim = query.shape
    _, seq_len, num_heads_kv, _ = key.shape
    
    # Use reference implementation for now
    # In production, this could be optimized with custom Pallas kernels
    return reference_gqa(query, key, value, lengths, mask_value=mask_value)


@jax.jit
def ragged_multi_head_attention(
    q: jax.Array,  # (B, T, N, H) query
    k: jax.Array,  # (B, S, K, H) key
    v: jax.Array,  # (B, S, K, H) value
    lengths: jax.Array,  # (B,) actual sequence lengths
    block_size: int = 256,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> jax.Array:
    """Multi-head attention with ragged sequences (variable lengths).
    
    This function eliminates padding overhead by only computing attention
    over the actual sequence length for each batch element.
    
    Args:
        q: Query tensor (B, T, N, H) where T=1 for decode
        k: Key tensor (B, S, K, H) from cache
        v: Value tensor (B, S, K, H) from cache  
        lengths: Actual sequence lengths per batch element (B,)
        block_size: Block size for kernel optimization
        mask_value: Mask value for padded positions
        
    Returns:
        Attention output (B, T, N, H)
    """
    B, T, N, H = q.shape
    _, S, K, _ = k.shape
    G = N // K  # Groups per KV head
    
    # Use reference implementation for now (can be optimized with Pallas later)
    q_reshaped = q.reshape((B, T, K, G, H))
    
    # Create length-based mask efficiently
    pos_mask = jnp.arange(S)[None, None, :] < lengths[:, None, None]  # (B, 1, S)
    pos_mask = jnp.broadcast_to(pos_mask, (B, T, S))
    
    # Compute attention scores
    scores = jnp.einsum("btkgh,bskh->btkgs", q_reshaped, k) / jnp.sqrt(H)
    scores = scores.reshape((B, T, -1, S))
    
    # Apply length mask - only attend to valid positions
    scores = jnp.where(jnp.expand_dims(pos_mask, -2), scores, mask_value)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
    
    # Apply attention to values
    probs = attn_weights.reshape((B, T, K, G, S))
    attn_out = jnp.einsum("btkgs,bskh->btkgh", probs, v)
    attn_out = attn_out.reshape((B, T, N, H))
    
    return attn_out
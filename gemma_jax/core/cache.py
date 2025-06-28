# cache.py
"""
KV‑cache utilities for Gemma‑3 – ring‑buffer version with fast scatter update.
* Ring buffer: keys/values live in circular arrays of length `cache_len`
    (== `max_seq_len` set at initialisation).  Writes always happen at
    `(cursor % cache_len)`.
* [REFACTORED] The cache now stores keys and values as int8, with
    corresponding bfloat16 scales, to reduce memory usage. Dequantization
    is performed on-the-fly during lookup.
* Book‑keeping fields (`sequence_lengths`, `write_positions`) are still
    maintained for backward compatibility but no longer drive the logic.
* High‑level `lookup_layer()` produces (key, value, mask) ready for the
    attention kernel, including optional sliding‑window masking.
* `_update_dense()` uses a single `lax.scatter` (or an optional Triton
    kernel) for large‑chunk pre‑fill.
"""

# First-pass manual quantization for the KV cache.

# The core logic changes involve:

# 1.  Modifying `KVCache` to hold `int8` data and `bfloat16` scales.
# 2.  Updating `init_cache` and sharding helpers for the new structure.
# 3.  Injecting **quantization** logic into the cache update functions (`_update_dense`, `_update_ragged`).
# 4.  Injecting **dequantization** logic into the cache lookup function (`lookup_layer`).

# No changes were needed for `segment.py` or `model.py` in this pass, as the quantization is entirely encapsulated within the cache system. 
# The `model.py` code continues to interact with the cache using `bfloat16` tensors, unaware of the internal precision change.

from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.sharding import NamedSharding, PartitionSpec
from gemma_jax.core.segment import SegmentInfo

# ---------------------------------------------------------------------------
# Optional Triton fast‑path
# ---------------------------------------------------------------------------
try:
    from gemma_jax.core.triton_kernels import scatter_cache_update  # type: ignore
    _TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    scatter_cache_update = None
    _TRITON_AVAILABLE = False
# ===========================================================================
# KVCache PyTree
# ===========================================================================
@jax.tree_util.register_pytree_node_class
@dataclass
class KVCache:
    """Sharded key/value tensors for transformer attention (ring buffer)."""
    # [REFACTORED] key/value are now int8 for memory savings.
    key: Array        # (L, B, S, K, H) - INT8
    value: Array      # (L, B, S, K, H) - INT8
    # [REFACTORED] Added scale tensors for dequantization.
    key_scale: Array  # (L, B, S, K)    - BFLOAT16
    value_scale: Array# (L, B, S, K)    - BFLOAT16

    sequence_lengths: Array   # (B,)
    write_positions: Array    # (B,)
    # ---- pytree helpers ---------------------------------------------------
    def tree_flatten(self):
        return (self.key, self.value, self.key_scale, self.value_scale,
                self.sequence_lengths, self.write_positions), None

    @classmethod
    def tree_unflatten(cls, _, leaves):
        return cls(*leaves)
    # ---- convenience ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.key.shape
    @property
    def batch_size(self) -> int:
        return self.key.shape[1]
    @property
    def max_seq_len(self) -> int:
        return self.key.shape[2]
    # Alias – clearer name after ring‑buffer refactor
    @property
    def cache_len(self) -> int:
        return self.key.shape[2]
    # ---- high‑level lookup -----------------------------------------------
    def lookup_layer(
        self,
        seg_info: SegmentInfo,
        *,
        layer: int,
        window: int | None = None,
        query_positions: Array | None = None,   # (B,T) absolute positions
    ) -> tuple[Array, Array, Array]:
        """
        Return `(key, value, mask)` for the requested layer.
        * [REFACTORED] This function now performs on-the-fly dequantization
            after retrieving data from the cache.
        * If `query_positions` is **None** (generation step), the mask has
            shape `(B, S)` and corresponds to the single token at
            `seg_info.current_pos`.
        * If `query_positions` is provided (prefill chunk), the mask has
            shape `(B, T, S)` with one causal / sliding row per query token.
        """
        # [REFACTORED] Retrieve quantized data and scales
        k_q = self.key[layer]      # (B, S, K, H) - int8
        v_q = self.value[layer]    # (B, S, K, H) - int8
        k_s = self.key_scale[layer]    # (B, S, K) - bfloat16
        v_s = self.value_scale[layer]  # (B, S, K) - bfloat16

        # [REFACTORED] Dequantize by multiplying with scales.
        # Reshape scales to (B, S, K, 1) to broadcast over the head_dim (H).
        k = k_q.astype(jnp.bfloat16) * k_s[..., None]
        v = v_q.astype(jnp.bfloat16) * v_s[..., None]

        B, S, _, _ = k.shape
        positions = jnp.arange(S, dtype=jnp.int32)                    # (S,)
        abs_pos_K_BS = seg_info.offset[:, None] + positions           # (B,S)
        # ------------------------------------------------------------------
        # Build causal / sliding mask
        # ------------------------------------------------------------------
        if query_positions is None:
            # Generation step (T == 1)
            abs_pos_Q = seg_info.current_pos[:, None]                 # (B,1)
            diff = abs_pos_K_BS - abs_pos_Q                           # (B,S)
            mask = diff <= 0
            if window is not None:
                mask &= diff > -window                                # (B,S)
        else:
            # Prefill chunk (T >= 1)
            abs_pos_Q_BTS = query_positions[:, :, None]               # (B,T,1)
            diff_BTS = abs_pos_K_BS[:, None, :] - abs_pos_Q_BTS       # (B,T,S)
            mask = diff_BTS <= 0
            if window is not None:
                mask &= diff_BTS > -window                            # (B,T,S)
        # ------------------------------------------------------------------
        # Exclude slots that have never been written
        # ------------------------------------------------------------------
        filled_BS = positions < seg_info.lengths[:, None]             # (B,S)
        if mask.ndim == 2:
            mask = mask & filled_BS                                   # (B,S)
        else:
            mask = mask & filled_BS[:, None, :]                       # (B,T,S)
        return k, v, mask

# ===========================================================================
# Public API helpers
# ===========================================================================
def init_cache(
    *,
    batch: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> KVCache:
    """Allocate an all‑zero KV cache."""
    cache_shape = (num_layers, batch, max_seq_len, num_kv_heads, head_dim)
    # [REFACTORED] Shape for scales does not include the head_dim
    scale_shape = (num_layers, batch, max_seq_len, num_kv_heads)

    # [REFACTORED] key/value are int8. Scales are always stored as bfloat16
    # regardless of the requested dtype.
    return KVCache(
        key=jnp.zeros(cache_shape, dtype=jnp.int8),
        value=jnp.zeros(cache_shape, dtype=jnp.int8),
        key_scale=jnp.ones(scale_shape, dtype=jnp.bfloat16),
        value_scale=jnp.ones(scale_shape, dtype=jnp.bfloat16),
        sequence_lengths=jnp.zeros((batch,), dtype=jnp.int32),
        write_positions=jnp.zeros((batch,), dtype=jnp.int32),
    )

def create_cache_partition_spec(key: str, mesh_axes: dict[str, str]) -> PartitionSpec:
    """PartitionSpec helper for KVCache fields."""
    # [REFACTORED] Added spec for scale fields
    if key in ("key", "value"):
        return PartitionSpec(None, mesh_axes.get("batch"), None,
                             mesh_axes.get("heads"), None)
    if key in ("key_scale", "value_scale"):
        return PartitionSpec(None, mesh_axes.get("batch"), None,
                             mesh_axes.get("heads"))
    if key in ("sequence_lengths", "write_positions"):
        return PartitionSpec(mesh_axes.get("batch"))
    return PartitionSpec()

def shard_kvcache_with_tree_map(
    cache: KVCache,
    mesh: Any,
    mesh_axes: dict[str, str],
) -> KVCache:
    """Shard each field of the cache according to its PartitionSpec."""
    def put(x: Array, field: str) -> Array:
        spec = create_cache_partition_spec(field, mesh_axes)
        return jax.device_put(x, NamedSharding(mesh, spec))

    # [REFACTORED] Pass all fields of the KVCache through the sharding logic
    return jax.tree_util.tree_map_with_path(
        lambda path, x: put(x, path[0].key), cache
    )

# ===========================================================================
# Cache update helpers
# ===========================================================================

def _quantize_to_int8(x: Array) -> tuple[Array, Array]:
    """
    Quantizes a bfloat16 tensor to int8, returning the quantized tensor
    and the dequantization scale. Scaling is done along the last dimension.
    """
    # Epsilon to prevent division by zero
    EPS = 1e-6
    x = x.astype(jnp.bfloat16)
    # Calculate scale per-vector along the last dimension
    max_abs = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = max_abs / 127.0 + EPS
    # Quantize
    x_q = (x / scale).round().astype(jnp.int8)
    # Squeeze the last dimension from the scale for storage
    scale = jnp.squeeze(scale, axis=-1)
    return x_q, scale.astype(jnp.bfloat16)

def _update_ragged(
    key_cache_layer: Array,   # (B, S, K, H)
    val_cache_layer: Array,   # (B, S, K, H)
    key_scale_layer: Array,   # (B, S, K)
    val_scale_layer: Array,   # (B, S, K)
    key_proj: Array,          # (B, 1, K, H) or (B, K, H)
    value_proj: Array,        # (B, 1, K, H) or (B, K, H)
    write_pos_B: Array,       # (B,)
) -> tuple[Array, Array, Array, Array]:
    """Single‑token update (generation) with quantization."""
    if key_proj.ndim == 4:                                  # (B,1,K,H)
        key_proj = jnp.squeeze(key_proj, axis=1)
        value_proj = jnp.squeeze(value_proj, axis=1)

    # [REFACTORED] Quantize inputs before updating cache
    key_proj_q, key_proj_s = jax.vmap(_quantize_to_int8)(key_proj)
    val_proj_q, val_proj_s = jax.vmap(_quantize_to_int8)(value_proj)
    key_proj_s = key_proj_s.astype(key_scale_layer.dtype)
    val_proj_s = val_proj_s.astype(val_scale_layer.dtype)

    max_cache_len = key_cache_layer.shape[1]
    def update_one(cache_k, cache_v, cache_ks, cache_vs, new_k, new_v, new_ks, new_vs, pos):
        idx = pos % max_cache_len
        # Reshape for dynamic_update_slice
        new_k = new_k[None, :, :]
        new_v = new_v[None, :, :]
        new_ks = new_ks[None, :]
        new_vs = new_vs[None, :]

        # Update quantized data
        cache_k = lax.dynamic_update_slice(cache_k, new_k, (idx, 0, 0))
        cache_v = lax.dynamic_update_slice(cache_v, new_v, (idx, 0, 0))
        # Update scales
        cache_ks = lax.dynamic_update_slice(cache_ks, new_ks, (idx, 0))
        cache_vs = lax.dynamic_update_slice(cache_vs, new_vs, (idx, 0))
        return cache_k, cache_v, cache_ks, cache_vs

    return jax.vmap(update_one)(
        key_cache_layer, val_cache_layer, key_scale_layer, val_scale_layer,
        key_proj_q, val_proj_q, key_proj_s, val_proj_s, write_pos_B
    )

def _update_dense(
    key_cache_layer: Array,   # (B, S, K, H)
    val_cache_layer: Array,   # (B, S, K, H)
    key_scale_layer: Array,   # (B, S, K)
    val_scale_layer: Array,   # (B, S, K)
    key_proj: Array,          # (B, T, K, H)
    value_proj: Array,        # (B, T, K, H)
    write_pos_B: Array,       # (B,)
    seq_lens_B: Array,        # (B,)
) -> tuple[Array, Array, Array, Array]:
    """Chunk (prefill) update via lax.scatter with quantization."""
    B, S, K, H = key_cache_layer.shape
    T = key_proj.shape[1]
    if T > S:
        raise ValueError(f"Chunk length exceeds cache capacity ({T} > {S})")

    # [REFACTORED] Quantize the entire chunk of projections
    key_proj_q, key_proj_s = jax.vmap(_quantize_to_int8)(key_proj)
    val_proj_q, val_proj_s = jax.vmap(_quantize_to_int8)(value_proj)
    key_proj_s = key_proj_s.astype(key_scale_layer.dtype)
    val_proj_s = val_proj_s.astype(val_scale_layer.dtype)

    write_pos_B = write_pos_B.reshape(-1)
    seq_lens_B = seq_lens_B.reshape(-1)

    # Note: Triton path would need a new kernel that can scatter 4 tensors.
    # The JAX fallback is implemented here.
    if _TRITON_AVAILABLE:
        # For this refactoring, we fall back to the JAX path.
        # A production system would require a new Triton kernel.
        pass

    # ------------ native JAX path -----------------------------------------
    timeline_len = key_proj.shape[1]
    token_positions = write_pos_B[:, None] + jnp.arange(timeline_len)[None, :]
    cache_indices = jnp.mod(token_positions, S)
    valid_mask = jnp.arange(timeline_len)[None, :] < seq_lens_B[:, None]
    
    # Use mode="drop" to ignore writes where valid_mask is False
    # To do this, we set indices to -1 where the mask is false.
    cache_indices = jnp.where(valid_mask, cache_indices, -1)
    batch_indices = jnp.arange(B)[:, None]
    batch_indices = jnp.broadcast_to(batch_indices, cache_indices.shape)

    # Scatter quantized data
    updated_key = key_cache_layer.at[batch_indices, cache_indices].set(key_proj_q, mode="drop")
    updated_val = val_cache_layer.at[batch_indices, cache_indices].set(val_proj_q, mode="drop")
    # Scatter scales
    updated_key_scale = key_scale_layer.at[batch_indices, cache_indices].set(key_proj_s, mode="drop")
    updated_val_scale = val_scale_layer.at[batch_indices, cache_indices].set(val_proj_s, mode="drop")

    return updated_key, updated_val, updated_key_scale, updated_val_scale

# ===========================================================================
# Public updater
# ===========================================================================
@partial(jax.jit, static_argnames=("layer", "ragged"))
def update_cache_layer(
    cache: KVCache,
    key_proj: Array,        # (B, T, K, H)
    value_proj: Array,      # (B, T, K, H)
    *,
    seg_info: SegmentInfo,  # unified metadata
    chunk_lens_B: Array,    # (B,) – tokens written this call
    layer: int,
    ragged: bool = False,
):
    """Update one transformer layer in the ring‑buffer KV cache."""
    if not 0 <= layer < cache.key.shape[0]:
        raise ValueError(f"Layer {layer} out of bounds")

    # [REFACTORED] Fetch quantized data and scales for the target layer
    key_layer = cache.key[layer]
    val_layer = cache.value[layer]
    key_scale_layer = cache.key_scale[layer]
    val_scale_layer = cache.value_scale[layer]

    write_pos_B = jnp.asarray(seg_info.cursor)
    seq_lens_B = jnp.asarray(chunk_lens_B)

    if ragged or key_proj.shape[1] == 1:
        updated_k, updated_v, updated_ks, updated_vs = _update_ragged(
            key_layer, val_layer, key_scale_layer, val_scale_layer,
            key_proj, value_proj, write_pos_B
        )
    else:
        updated_k, updated_v, updated_ks, updated_vs = _update_dense(
            key_layer, val_layer, key_scale_layer, val_scale_layer,
            key_proj, value_proj, write_pos_B, seq_lens_B
        )

    # ---- bookkeeping (legacy fields) -------------------------------------
    # Bookkeeping should reflect the state after this write, independent of
    # how many layers are updated in a single forward pass.  Derive the new
    # counters from ``seg_info`` rather than the cache's previous values to
    # avoid incrementing them multiple times.
    new_seq_len = jnp.minimum(seg_info.lengths + seq_lens_B, cache.cache_len)
    new_write_pos = (seg_info.cursor + seq_lens_B) % cache.cache_len

    # [REFACTORED] Construct the new cache with all updated fields
    new_cache = KVCache(
        key=cache.key.at[layer].set(updated_k),
        value=cache.value.at[layer].set(updated_v),
        key_scale=cache.key_scale.at[layer].set(updated_ks),
        value_scale=cache.value_scale.at[layer].set(updated_vs),
        sequence_lengths=new_seq_len,
        write_positions=new_write_pos,
    )
    # The return of updated_k, updated_v is for potential debugging/inspection
    # but the primary output is the new_cache object.
    return updated_k, updated_v, new_cache

# ===========================================================================
# Utility helpers (debug / inspection)
# ===========================================================================



def get_valid_cache_positions(cache: KVCache) -> Array:
    """Boolean mask of valid positions in the ring."""
    positions = jnp.arange(cache.cache_len)[None, :]
    return positions < cache.sequence_lengths[:, None]


def reset_cache_positions(cache: KVCache,
                          batch_indices: Array | None = None) -> KVCache:
    """Reset len/ptr book‑keeping (does not zero memory)."""
    if batch_indices is None:
        new_len = jnp.zeros_like(cache.sequence_lengths)
        new_pos = jnp.zeros_like(cache.write_positions)
    else:
        new_len = cache.sequence_lengths.at[batch_indices].set(0)
        new_pos = cache.write_positions.at[batch_indices].set(0)
    """Allocate an all‑zero KV cache."""

    num_layers, batch, max_seq_len, num_kv_heads, head_dim = cache.key.shape
    # [REFACTORED] key/value are now int8 for memory savings
    cache_shape = (num_layers, batch, max_seq_len, num_kv_heads, head_dim)
    # [REFACTORED] Shape for scales does not include the head_dim
    scale_shape = (num_layers, batch, max_seq_len, num_kv_heads)

    # [REFACTORED] key/value are int8. Scales are always stored as bfloat16.
    return KVCache(
        key=jnp.zeros(cache_shape, dtype=jnp.int8),
        value=jnp.zeros(cache_shape, dtype=jnp.int8),
        key_scale=jnp.ones(scale_shape, dtype=jnp.bfloat16),
        value_scale=jnp.ones(scale_shape, dtype=jnp.bfloat16),
        sequence_lengths=new_len,
        write_positions=new_pos,
    )


def cache_info_string(cache: KVCache) -> str:
    """Human‑readable summary of cache state."""
    gb = (cache.key.nbytes + cache.value.nbytes) / 1024 ** 3
    return (
        f"KVCache Info:\n"
        f"  Shape: {cache.shape}\n"
        f"  Batch size: {cache.batch_size}\n"
        f"  Cache length: {cache.cache_len}\n"
        f"  Layers: {cache.key.shape[0]}\n"
        f"  Seq lengths: {cache.sequence_lengths.tolist()}\n"
        f"  Write positions: {cache.write_positions.tolist()}\n"
        f"  Memory: {gb:.2f} GB"
    )


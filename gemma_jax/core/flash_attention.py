"""Wrapper for JetStream flash attention."""

from __future__ import annotations

import logging

try:
    from jax.experimental import jetstream as js
    HAS_JETSTREAM = True
except Exception as e:  # pragma: no cover - optional dependency
    js = None
    HAS_JETSTREAM = False
    logging.warning("JetStream flash attention not available: %s", e)

try:
    import jax
    import jax.numpy as jnp
    from jax import Array
except Exception as e:  # pragma: no cover - missing deps
    jax = None  # type: ignore
    jnp = None  # type: ignore
    Array = object  # type: ignore
    logging.warning("JAX not available: %s", e)


def flash_attention(
    q: Array,
    k: Array,
    v: Array,
    attn_mask_BTS: Array,
) -> Array:
    """Flash attention wrapper with ``multi_head_attention`` interface."""
    if not HAS_JETSTREAM or jax is None:
        raise RuntimeError("JetStream flash attention kernel not available")

    # JetStream expects mask of shape (B, T, S) with True where attention allowed.
    out = js.flash_attention(q, k, v, attn_mask_BTS)
    return out

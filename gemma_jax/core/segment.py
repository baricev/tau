# gemma_jax/core/segment.py
"""Light‑weight metadata object for causal/streaming generation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SegmentInfo:
    """Book‑keeping for each batch element during generation.

    Args:
      lengths:  Valid tokens that already reside in the KV‑cache  – shape (B,)
      cursor:   Next physical write index (ring‑buffer position)  – shape (B,)
      offset:   Absolute position of cache slot 0                – shape (B,)
      cache_len: Length of the ring (== max_seq_len today)
    """

    lengths: Array        # (B,)
    cursor:  Array        # (B,)
    offset:  Array        # (B,)
    cache_len: int

    # ── pytree helpers ──────────────────────────────────────────────────────
    def tree_flatten(self):
        return (self.lengths, self.cursor, self.offset), self.cache_len

    @classmethod
    def tree_unflatten(cls, cache_len: Any, leaves):
        lengths, cursor, offset = leaves
        return cls(lengths, cursor, offset, cache_len)

    # ── convenience ---------------------------------------------------------
    @property
    def current_pos(self) -> Array:
        """Absolute position of the *token being processed* (cursor – 1)."""
        return self.offset + (self.cursor - 1)

    @property
    def next_pos(self) -> Array:
        """Absolute position of the slot the *next* token will occupy."""
        return self.offset + self.cursor

    def advance(self, n_tokens: int | Array = 1) -> "SegmentInfo":

        """Return a new `SegmentInfo` after writing `n_tokens`, with wrap‑aware logic."""
        n_tokens = jnp.asarray(n_tokens, dtype=self.lengths.dtype)

        prev_len   = self.lengths
        new_len    = jnp.minimum(prev_len + n_tokens, self.cache_len)

        new_cursor = (self.cursor + n_tokens) % self.cache_len

        # Extra tokens that caused a wrap (i.e. the amount of oldest context discarded)
        overflow   = jnp.maximum(prev_len + n_tokens - self.cache_len, 0)
        new_offset = self.offset + overflow

        return SegmentInfo(new_len, new_cursor, new_offset, self.cache_len)
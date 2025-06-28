import jax
import jax.numpy as jnp
from gemma_jax.core.cache import init_cache
from gemma_jax.core.segment import SegmentInfo


def test_lookup_layer_zero_window():
    cache = init_cache(batch=1, max_seq_len=4, num_layers=1,
                       num_kv_heads=1, head_dim=1, dtype=jnp.bfloat16)
    seg = SegmentInfo(jnp.array([2], jnp.int32), jnp.array([2], jnp.int32),
                      jnp.array([0], jnp.int32), cache_len=4)
    _, _, mask_none = cache.lookup_layer(seg, layer=0, window=None)
    _, _, mask_zero = cache.lookup_layer(seg, layer=0, window=0)
    assert mask_zero.sum() == 0
    assert mask_none.sum() > 0

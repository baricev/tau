import jax
import jax.numpy as jnp
from gemma_jax.core.cache import init_cache, update_cache_layer
from gemma_jax.core.segment import SegmentInfo


def test_update_cache_single_step():
    cache = init_cache(batch=1, max_seq_len=4, num_layers=2,
                       num_kv_heads=1, head_dim=1, dtype=jnp.bfloat16)
    seg = SegmentInfo(jnp.array([0], jnp.int32), jnp.array([0], jnp.int32),
                      jnp.array([0], jnp.int32), cache_len=4)
    key_proj = jnp.ones((1, 1, 1, 1), dtype=jnp.bfloat16)
    val_proj = key_proj
    chunk = jnp.array([1], jnp.int32)
    for layer in range(2):
        _, _, cache = update_cache_layer(cache, key_proj, val_proj,
                                        seg_info=seg, chunk_lens_B=chunk,
                                        layer=layer, ragged=True)
    assert int(cache.write_positions[0]) == 1

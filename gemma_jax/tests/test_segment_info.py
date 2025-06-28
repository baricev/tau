import jax
import jax.numpy as jnp
from gemma_jax.core.model import setup_scan_fn
from gemma_jax.core.cache import init_cache


def test_setup_scan_fn_cursor():
    batch = 1
    cache = init_cache(batch=batch, max_seq_len=4, num_layers=1,
                       num_kv_heads=1, head_dim=1, dtype=jnp.bfloat16)
    input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    state = None
    last_tok, seg_info, step, _, _ = setup_scan_fn(state, input_ids, cache)
    assert int(seg_info.cursor[0]) == 2
    assert int(seg_info.current_pos[0]) == 1

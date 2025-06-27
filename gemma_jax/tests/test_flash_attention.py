import jax
import jax.numpy as jnp
from gemma_jax.core.model import multi_head_attention
from gemma_jax.core.flash_attention import multi_head_flash_attention


def test_flash_attention_matches_reference():
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (2, 3, 4, 8), dtype=jnp.bfloat16)
    k = jax.random.normal(key, (2, 3, 2, 8), dtype=jnp.bfloat16)
    v = jax.random.normal(key, (2, 3, 2, 8), dtype=jnp.bfloat16)
    mask = jnp.ones((2, 3, 3), dtype=bool)

    ref = multi_head_attention(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32), mask)
    flash = multi_head_flash_attention(q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32), mask)
    assert jnp.allclose(ref, flash, atol=1e-2)

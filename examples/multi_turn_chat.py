# %% [markdown]
# ## Gemma JAX: Multi-Turn Batched Inference Notebook
#
# This notebook demonstrates how to perform batched, multi-turn inference using the Gemma JAX model.

# %% [markdown]
# ### Imports and Environment Setup
#
# Conditionally installs dependencies and sets up the environment based on whether it's running locally or in Google Colab.

# %%

import time
import numpy as np
import jax
from jax import numpy as jnp

# Import all necessary objects and functions from setup.py
from setup import (
    model,
    tokenizer,
    config,
    state,
    cache,
    rope_cache,
    encode_text,
    chunked_prefill,
    setup_carry,
    paxml_generate_chunked_scan_queue,
    run_full_generation_with_chunked_callback,
    reset_threading_state,
    _CHUNK_QUEUE,
    _consumer_t,
    _GENERATION_COMPLETE,
    _CONSUMER_STARTED,
    _EXPECTED_CHUNKS,
    _PROCESSED_CHUNKS,
    _CHUNKS_LOCK
)

# Import additional needed functions for display
from gemma_jax.core.sp_tokenizer import format_prompt

PAD_ID, EOS_ID, BOS_ID, END_OF_TURN_ID = 0, 1, 2, 106

# %% [markdown]
# ### Model Objects
#
# All model objects (model, tokenizer, config, state, cache, rope_cache) are imported from setup.py
# %%

# Model objects are already loaded from setup.py
print("Model configuration:")
print(f"batch_size={config.batch_size}, cache_length={config.cache_length}, chunk_length={config.chunk_length}")
print(f"window_size={config.window_size}, generate_steps={config.generate_steps}")
print(config)

# Reset threading state to avoid conflicts from setup.py import
reset_threading_state()


# %%
# Example usage of the imported model objects

prompts = [
    "I love to",
    "Explain general relativity to a first-century Roman philosopher (Cicero).",
]

from gemma_jax.tests.prompts import p1, p2, p3
long_prompts = [
    p1,
    p2[:3000] + "\n\nSummarize the text above in a few sentences.",
    p3[:3000] + "\n\nSummarize the text above in a few sentences.",
    "Explain general relativity to a first-century Roman philosopher (Cicero)."
]

# Use prompts that match the batch size
prompts = (long_prompts * config.batch_size)[:config.batch_size]
formatted_prompts = [format_prompt(p) for p in prompts]

input_ids = encode_text(formatted_prompts, tokenizer, add_special_tokens=True, return_tensors="np")
print("input ids:", input_ids.shape, ", seq lens:", (input_ids != 0).sum(axis=-1), "chunk_length:", config.chunk_length)

padded_input_ids = np.pad(input_ids, ((0, 0), (0, config.cache_length - input_ids.shape[1])), constant_values=0)
print(f"padded input ids: {padded_input_ids.shape}, seq lens:{(padded_input_ids != 0).sum(axis=-1)}")

jax_padded_input_ids = jnp.array(padded_input_ids, dtype=jnp.int32)  # Convert to JAX array with static shape

print(f"JAX static input ids: {jax_padded_input_ids.shape}, seq lens:{(jax_padded_input_ids != 0).sum(axis=-1)}")

# %% [markdown]
# ### Inference: Prefill Stage
#
# Perform prefill inference. The cache is populated with embeddings from the input sequences, preparing it for autoregressive generation.
# %%


t0 = time.time()
prefill_cache = chunked_prefill(
    state=state,
    input_ids=jax_padded_input_ids,
    model=model,
    cache=cache,
    rope_cache=rope_cache,
    config=config,
    chunk_length=config.chunk_length,
    cache_length=config.cache_length,
)
print(f"Prefill completed in {time.time() - t0:.2f}s")

t0 = time.time()

# init_carry using imported function
init_carry = setup_carry(state=state, input_ids=input_ids, prefill_cache=prefill_cache)
print(f"Setup generate completed in {time.time() - t0:.2f}s")

# Time the generation using imported function
t0 = time.time()
total_tokens = config.generate_steps # Total number of tokens to generate
run_chunk_size = config.chunk_length # Number of tokens to generate in each chunk
_ = run_full_generation_with_chunked_callback(
    init_carry,
    model=model,
    rope_cache=rope_cache,
    config=config,
    tokenizer_arg=tokenizer,
    total_tokens=total_tokens,
    chunk_size=run_chunk_size
)

dt = time.time() - t0
print(f"Generation completed in {dt:.2f}s")
print(f"Generated {total_tokens * config.batch_size} tokens in {dt:.2f}s, {total_tokens * config.batch_size / dt:.2f} tokens/sec")

# Clean shutdown using imported variables
if _consumer_t and _consumer_t.is_alive():
    # Send poison pill to stop consumer thread
    _CHUNK_QUEUE.put(None)
    # Wait for thread to finish
    _consumer_t.join(timeout=0.1)
    if _consumer_t.is_alive():
        print("Warning: Consumer thread did not exit cleanly")

print("Script completed successfully!")

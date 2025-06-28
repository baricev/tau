# %% 
# setup.py

import argparse
import os
import sys
import queue
import threading

import time
from functools import partial
import math
from pathlib import Path
from typing import Any

import numpy as np
import jax
from jax import numpy as jnp
from jax import Array

from gemma_jax.core.weights import create_config  as original_create_gemma3_config
from gemma_jax.core.weights import create_device_mesh, load_model, load_unsharded_model
from gemma_jax.core.rope import init_rope_cache, load_rope_cache
from gemma_jax.core.sp_tokenizer import SentencePieceTokenizer, encode_raw_ids, process_and_pad_inputs, encode_text, decode_tokens, format_prompt
from gemma_jax.core.inference import greedy_sample

# from gemma_jax.core.chunked_prefill import _scatter_token, _gather_token

from gemma_jax.core.cache import init_cache, create_cache_partition_spec, shard_kvcache_with_tree_map, KVCache
from gemma_jax.core.model import forward_fn, decode
from gemma_jax.core.segment import SegmentInfo

from jax.experimental.pallas.ops.tpu import flash_attention, splash_attention

PAD_ID, EOS_ID, BOS_ID, END_OF_TURN_ID = 0, 1, 2, 106

def in_notebook():
    """Check if running in a Jupyter notebook (including VS Code)."""
    try:
        # Check if IPython is available and we're in an interactive shell
        from IPython import get_ipython
        if get_ipython() is not None:
            # Check if it's a notebook kernel (ZMQ) vs terminal IPython
            return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        pass
    return False

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_repo_root():
    """Get the root directory of the gemma-jax repository."""
    from pathlib import Path
    try:
      return Path(__file__).parent.parent
    except NameError:
      return Path.cwd()


def _parse_arguments():
    """
    Parses command-line arguments for paths, safely handling notebook environments.

    This is a helper function designed not to be called directly by the user.
    """
    parser = argparse.ArgumentParser(
        description="Provide optional paths for a model checkpoint and tokenizer."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the model checkpoint file. If not provided, a default will be used."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer file. If not provided, a default will be used."
    )

    # Detect if the script is running in an interactive notebook (like Colab or Jupyter).
    # The kernel passes its own arguments, which we want to ignore.
    if any('colab_kernel_launcher' in arg for arg in sys.argv) or any('ipykernel_launcher' in arg for arg in sys.argv):
        # In a notebook, parse an empty list of arguments to avoid conflicts.
        # This makes argparse use the default values we defined (which is None).
        print("INFO: Running in a notebook environment. Ignoring command-line arguments.")
        args = parser.parse_args([])
    else:
        # When run from a terminal, parse the arguments as usual.
        args = parser.parse_args()

    return args

def maybe_get_paths_from_args(default_checkpoint_path: Path, default_tokenizer_path: Path) -> tuple[Path, Path]:
    """
    Gets checkpoint and tokenizer paths, prioritizing command-line args over defaults.

    This function can be safely called from a script or an interactive notebook cell.

    Args:
        default_checkpoint_path: The default path to use if no checkpoint is provided via CLI.
        default_tokenizer_path: The default path to use if no tokenizer is provided via CLI.

    Returns:
        A tuple containing the final (Path, Path) for the checkpoint and tokenizer.
    """
    # 1. Get arguments from the command line (or None if in a notebook/not provided)
    cli_args = _parse_arguments()

    # 2. Decide which path to use. Prioritize the one from the command line.
    checkpoint_path = Path(cli_args.checkpoint_path) if cli_args.checkpoint_path else default_checkpoint_path
    tokenizer_path = Path(cli_args.tokenizer_path) if cli_args.tokenizer_path else default_tokenizer_path

    return checkpoint_path, tokenizer_path


# %% [markdown]
# ### Configuration Defaults
#
# Set the default paths and parameters. Adjust the paths to reflect your actual filesystem setup for the tokenizer and model checkpoints.

# %%
TOKENIZER_PATH = get_repo_root() / "tokenizer.model"
CHECKPOINT_PATH = get_repo_root() / "4b"

IN_NOTEBOOK = in_notebook()
IN_SCRIPT = not IN_NOTEBOOK
IN_COLAB = False

try:
  import google.colab
  IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    print(f"Running in Google Colab. Current directory: {os.getcwd()}")
    root_dir = Path('/content/gemma_jax')
    TOKENIZER_PATH =  root_dir / 'tokenizer.model'       # Absolute path to the Gemma model checkpoint
    CHECKPOINT_PATH =  Path("/content/drive/MyDrive/4b") # Absolute path to the Gemma model checkpoint

if IN_SCRIPT:

    try:
        print("Getting paths from args...")
        CHECKPOINT_PATH, TOKENIZER_PATH = maybe_get_paths_from_args(CHECKPOINT_PATH, TOKENIZER_PATH)
    except:
        pass

# Assert paths exist
assert TOKENIZER_PATH.exists(), f"Tokenizer path {TOKENIZER_PATH} does not exist."
assert CHECKPOINT_PATH.exists(), f"Checkpoint path {CHECKPOINT_PATH} does not exist."

# Assert paths are absolute
assert TOKENIZER_PATH.is_absolute(), f"Tokenizer path {TOKENIZER_PATH} is not absolute."
assert CHECKPOINT_PATH.is_absolute(), f"Checkpoint path {CHECKPOINT_PATH} is not absolute."

print(f"Tokenizer path: {TOKENIZER_PATH}")
print(f"Checkpoint path: {CHECKPOINT_PATH}")

# %% [markdown]
# ### Installation
#
# Install required Python packages for TPU support and dataset management.

# %%
# !pip install -e . --quiet

# %% 

# If running in Colab, mount the drive and clone the gemma-jax repository (uncomment the lines below)

# from google.colab import drive
# drive.mount("/content/drive")
# # Clone the gemma-jax repository if it doesn't exist

# if not os.path.exists('gemma-jax'):
#     print("Cloning gemma-jax repository...")

#     ! git clone https://github.com/baricev/gemma-jax

#     # Change the current working directory to the gemma-jax repository
#     os.chdir('/content/gemma-jax')

#     ! pip install -e . --quiet



# %% [markdown]
#
# ### Running the script directly from a Notebook (Colab or VS Code)
#  
# See commented out lines below for example commands
# %%  
# Colab:
# !python examples/multi_turn_chat.py --tokenizer_path /content/tau/tokenizer.model --checkpoint_path /content/drive/MyDrive/4b
# 
# VS Code:
# !python multi_turn_chat.py --tokenizer_path  /Users/v/new_workspace/baricev-gemma-jax-final-june-26/tokenizer.model
# 

# %% [markdown]
# ### Model Loading
#
# Load the model and tokenizer.
# %%

def make_model_objects(
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_size: int = 4,
    cache_length: int = 1024,
    chunk_length: int = 128,
    window_size: int = 1024,
    dtype_str: str = "bfloat16",
    batch_size: int = 2,
    generate_steps: int = 8,
    shard_model: bool = True,
):
    """
    Build the mesh, load weights, create caches, etc.  Matches the notebook.
    """

    mesh = create_device_mesh()
    config = original_create_gemma3_config(
        model_size=model_size,
        batch_size=batch_size,
        cache_length=cache_length,
        chunk_length=chunk_length,
        window_size=window_size,
        generate_steps=generate_steps,  # we override later
    )

    kv_cache = init_cache(
        batch=batch_size,
        max_seq_len=cache_length,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    mesh_axes = {"batch": "data", "heads": "model"}

    model_dtype = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[dtype_str]

    if shard_model:
        model = load_model(checkpoint_path, mesh, config, dtype=model_dtype)
        rope_cache = load_rope_cache(mesh, config)
        kv_cache = shard_kvcache_with_tree_map(kv_cache, mesh, mesh_axes)

    else:
        # Load the model without sharding, useful for single-device inference
        # or debugging purposes.
        print("Loading unsharded model...")
        model = load_unsharded_model(checkpoint_path, config, dtype=model_dtype)
        rope_cache = init_rope_cache(config)


    tokenizer = SentencePieceTokenizer(tokenizer_path)

    return model, tokenizer, rope_cache, kv_cache, config



# %% [markdown]
# ### Model Configuration
#
# Define model hyperparameters and inference settings, such as cache size, sequence lengths, batch size, and data types. Adjust these according to your computational resources and experimental needs.

# %%

start_setup = time.time()

model_size = 4  # Model scale (e.g., 4 for 4B parameters)
cache_length = 4096  # Length of KV-cache
chunk_length = 1024  # Maximum input sequence length
window_size = 1024  # Attention window size
batch = 4  # Batch size for inference
dtype_str = "bfloat16"  # Model precision: ['bfloat16', 'float16', 'float32']
generate_steps = 1024  # Number of tokens generated after prefill

# XLA # TODO: if needed, update mem_fraction
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# %% [markdown]
# ### Initialization and Model Loading
#
# This step includes initializing the device mesh, loading the Gemma model checkpoint, creating RoPE caches, initializing KV caches, and loading the tokenizer. 
# It dynamically adjusts settings based on the detected hardware (CPU or TPU).

if jax.devices()[0].device_kind == "cpu":
  print("Using CPU device settings.")
  cache_length, chunk_length, window_size,  batch, generate_steps = 128, 128, 128, 2, 8
  cache_length, chunk_length, window_size,  batch, generate_steps = 1*1024, 128, 1024, 4, 32 
  shard_model = False # True for TPU, False for CPU
  # os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
else:
  print("Using TPU device settings.")
  shard_model = True # True for TPU, False for CPU

model, tokenizer, rope_cache, cache, config = make_model_objects(
    checkpoint_path=CHECKPOINT_PATH,
    tokenizer_path=TOKENIZER_PATH,
    model_size=model_size,
    cache_length=cache_length,
    chunk_length=chunk_length,
    window_size=window_size,
    dtype_str=dtype_str,
    batch_size=batch,
    generate_steps=generate_steps,
    shard_model=shard_model,  # Set to False for debugging or single-device inference
)

# Dummy state for the model, can be used to store additional parameters or state
state = (jnp.array([42]))

print(f"Model {model_size}B loaded in {time.time() - start_setup:.2f}s")
# %%

@jax.jit
def setup_carry(*, state: Any, input_ids: Array, prefill_cache: Any) -> tuple:
    """Prepare the new SegmentInfo‑based carry."""
    B         = input_ids.shape[0]
    seq_lens  = (input_ids != 0).sum(axis=-1)                      # (B,)
    last_tok  = input_ids[jnp.arange(B), seq_lens - 1]             # (B,)

    seg_info = SegmentInfo(
        lengths=seq_lens,      # tokens already in cache
        cursor=seq_lens,       # next write slot
        offset=jnp.zeros_like(seq_lens),
        cache_len=int(prefill_cache.max_seq_len),
    )

    carry = (
        last_tok,              # 0 current input token (B,)
        seg_info,              # 1 SegmentInfo
        0,                     # 2 step counter
        prefill_cache,         # 3 KV‑cache
        state,                 # 4 model params/state
    )
    return carry


def make_chunk_mask(
    batch_size: int,
    start_idx: int,
    chunk_length: int,
    cache_length: int,
) -> Array:
  """
  Build a (B, T_chunk, cache_length) causal mask for a chunk length.

  For queries in positions  [start_idx, start_idx + chunk_length - 1]
  return True whenever a key index  k  is  l.t.e  current query pos.

  Return Shape:  (B, T_chunk, cache_length)
  """
  # (T_chunk, 1)
  q_pos = jnp.arange(chunk_length, dtype=jnp.int32)[:, None] + start_idx
  # (1, cache_length)
  k_pos = jnp.arange(cache_length, dtype=jnp.int32)[None, :]
  mask = k_pos <= q_pos  # (T_chunk, cache_length)
  mask = jnp.broadcast_to(mask, (batch_size, *mask.shape))
  return mask


# Helper: Build position ids
def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

# Chunked prefill implementation using static shapes
@partial(jax.jit, static_argnames=("config", "chunk_length", "cache_length"))
def chunked_prefill(
    state: Any,
    input_ids: jax.Array,  # (B, S_total)
    model: Any,
    cache: KVCache,
    rope_cache: Any,
    config: Any,
    *,
    chunk_length: int,
    cache_length: int,
) -> KVCache:
    """
    Process an arbitrarily long prompt in fixed-size chunks using static shapes.
    Returns the filled KV-cache.
    """

    batch_size, seq_len = input_ids.shape

    num_chunks = seq_len // chunk_length + (seq_len % chunk_length > 0)

    # Pad tokens to multiple of chunk_length so shapes remain static
    pad_len = num_chunks * chunk_length - seq_len
    padded_input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)

    # Compute absolute positions
    padded_attn_mask = padded_input_ids != 0
    padded_position_ids = build_positions(padded_attn_mask)

    seq_lens_B = padded_attn_mask.sum(axis=-1)

    def body(carry, idx):
        kv_cache, model_state = carry

        start = idx * chunk_length

        tok_chunk = jax.lax.dynamic_slice(
            padded_input_ids, (0, start), (batch_size, chunk_length)
        )
        pos_chunk = jax.lax.dynamic_slice(
            padded_position_ids, (0, start), (batch_size, chunk_length)
        )

        # Compute attention mask for chunk (B, chunk_length, cache_length)
        attn_mask_BTS = make_chunk_mask(
            batch_size, start_idx=start, chunk_length=chunk_length, cache_length=cache_length
        )

        write_idx_B = jnp.full((batch_size,), start, dtype=jnp.int32)

        # SegmentInfo describing the cache **before** this chunk is written
        seg_info = SegmentInfo(
            lengths=write_idx_B,          # prompt length so far   (B,)
            cursor=write_idx_B,           # write head starts here (B,)
            offset=jnp.zeros_like(write_idx_B),
            cache_len=int(cache_length),
        )

        _, updated_cache = forward_fn(
            model_state,
            tok_chunk,
            pos_chunk,
            # attn_mask_BTS,
            seg_info,                     #  NEW
            model=model,
            cache=kv_cache,
            rope_cache=rope_cache,
            config=config,
            auto_regressive=False,
            mesh=None,                    # keep None – single‑host
        )

        return (updated_cache, model_state), None

    (filled_cache, _), _ = jax.lax.scan(body, (cache, state), jnp.arange(num_chunks))

    return filled_cache




@partial(jax.jit, static_argnames=("config"))
def _generate_one_step(
    carry,
    *,
    model,
    rope_cache,
    config,
):
    """
    Single token autoregressive step (jitted).

    carry = (cur_tok, seq_len, write_idx, abs_pos, step, kv_cache, model_state)
    """
    # (cur_tok, seq_len, write_idx, abs_pos, step, kv_cache, model_state) = carry
    (cur_tok, seg_info, step, kv_cache, model_state) = carry

    batch   = cur_tok.shape[0]
    cache_L = config.cache_length

    # attn_mask = build_gen_step_attn_masks(
    #     seg_info.cursor[:, None],
    #     cache_L,
    #     jnp.ones((batch, cache_L), dtype=jnp.bool_),
    # )
    x_emb, updated_kv_cache = forward_fn(
        model_state,
        cur_tok[:, None],
        seg_info.next_pos[:, None],
        # attn_mask,
        seg_info,                         #  NEW
        model=model,
        cache=kv_cache,
        rope_cache=rope_cache,
        config=config,
        auto_regressive=True,
        mesh=None,
    )


    logits   = decode(model, x_emb).squeeze(axis=1)          # shape (B,1,V) -> (B, V)
    next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32) # shape (B,)

    new_carry = (
        next_tok,
        seg_info.advance(1),              #  NEW
        step + 1,
        updated_kv_cache,
        model_state,
    )
    return new_carry, next_tok


# =======================================================
# Queue-based chunked generation
# =======================================================

# A global queue for the chunk data
_CHUNK_QUEUE = queue.Queue()  # Use regular Queue instead of SimpleQueue

# Add a flag to track completion
_GENERATION_COMPLETE = threading.Event()
_CONSUMER_STARTED = threading.Event()

# Track expected chunks
_EXPECTED_CHUNKS = 0
_PROCESSED_CHUNKS = 0
_CHUNKS_LOCK = threading.Lock()

def consumer_thread(tokenizer_arg):
    """Consumer thread that processes chunks from the queue."""
    global _PROCESSED_CHUNKS
    _CONSUMER_STARTED.set()
    
    while True:
        try:
            # Use timeout to periodically check if we should exit
            item = _CHUNK_QUEUE.get(timeout=0.1)
        except queue.Empty:
            # Check if we're done
            if _GENERATION_COMPLETE.is_set():
                with _CHUNKS_LOCK:
                    if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS:
                        break
            continue
            
        if item is None:  # Poison pill
            break
            
        chunk = item["chunk"]  # shape (chunk_size, B)
        chunk_id = item["chunk_id"]
        
        # Decode the chunk
        batch_size = chunk.shape[1]
        unpermuted = chunk.reshape(-1, batch_size).transpose(1, 0)
        text = tokenizer_arg.batch_decode(unpermuted)
        
        print(f"[Consumer thread] chunk {chunk_id}, text='{text}'")
        
        with _CHUNKS_LOCK:
            _PROCESSED_CHUNKS += 1
        
        # Mark task as done
        _CHUNK_QUEUE.task_done()
    
    print("[Consumer thread] Exiting.")


# Don't start the thread yet - we'll start it when needed
_consumer_t = None


@partial(jax.jit, static_argnames=("config", "chunk_size", "chunk_id"))
def paxml_generate_chunked_scan_queue(
    init_carry,
    *,
    model,
    rope_cache,
    config,
    chunk_size: int,
    chunk_id: int,  # We'll pass a chunk_id so we know which chunk we're on
):
    """
    Single compiled function that does chunk_size steps in one scan,
    then returns the entire chunk. Minimizes callback overhead:
    - Just device->host copy
    - Then put data on a queue for async decode
    """

    def step_fn(carry, _):
        new_carry, next_tok = _generate_one_step(
            carry,
            model=model,
            rope_cache=rope_cache,
            config=config
        )
        return new_carry, next_tok  # shape (B,)

    final_carry, tokens_chunk = jax.lax.scan(
        step_fn,
        init_carry,
        xs=None,
        length=chunk_size
    )
    # tokens_chunk has shape (chunk_size, B)

    # Minimal callback: do a quick device->host copy, then enqueue
    def tap_chunk(dev_chunk):
        # Quick device->host (use jax.device_get for non-blocking transfer)
        chunk_host = jax.device_get(dev_chunk)  # shape (chunk_size, B)
        # Put on a queue for decoding
        _CHUNK_QUEUE.put({
            "chunk_id": chunk_id,
            "chunk": chunk_host
        })

    # Attach the callback
    jax.experimental.io_callback(
            tap_chunk,
            None,
            tokens_chunk,
            ordered=True)

    return final_carry, tokens_chunk


# =======================================================
# The driver that calls chunked scan repeatedly
# =======================================================

def reset_threading_state():
    """Reset threading state to avoid conflicts between multiple imports."""
    global _consumer_t, _EXPECTED_CHUNKS, _PROCESSED_CHUNKS
    
    # Clean up existing thread if running
    if _consumer_t and _consumer_t.is_alive():
        _CHUNK_QUEUE.put(None)  # Poison pill
        _consumer_t.join(timeout=1.0)
    
    # Reset state
    _consumer_t = None
    _GENERATION_COMPLETE.clear()
    _CONSUMER_STARTED.clear()
    with _CHUNKS_LOCK:
        _PROCESSED_CHUNKS = 0
        _EXPECTED_CHUNKS = 0
    
    # Clear queue
    while not _CHUNK_QUEUE.empty():
        try:
            _CHUNK_QUEUE.get_nowait()
        except queue.Empty:
            break

def run_full_generation_with_chunked_callback(
    init_carry,
    *,
    model,
    rope_cache,
    config,
    tokenizer_arg,
    total_tokens: int,
    chunk_size: int
):
    """
    Python driver that calls the chunked scan repeatedly until
    we've generated total_tokens.
    """
    global _consumer_t, _EXPECTED_CHUNKS, _PROCESSED_CHUNKS

    # Reset counters
    with _CHUNKS_LOCK:
        _PROCESSED_CHUNKS = 0
        _EXPECTED_CHUNKS = (total_tokens + chunk_size - 1) // chunk_size

    # Clear flags
    _GENERATION_COMPLETE.clear()
    _CONSUMER_STARTED.clear()

    # Start consumer thread if not already running
    if _consumer_t is None or not _consumer_t.is_alive():
        _consumer_t = threading.Thread(target=consumer_thread, args=(tokenizer_arg,))
        _consumer_t.start()
        # Wait for consumer to be ready
        _CONSUMER_STARTED.wait()

    carry = init_carry
    all_collected = []

    tokens_left = total_tokens
    chunk_count = 0

    while tokens_left > 0:
        current_chunk_size = min(tokens_left, chunk_size)
        carry, chunk_device = paxml_generate_chunked_scan_queue(
            carry,
            model=model,
            rope_cache=rope_cache,
            config=config,
            chunk_size=current_chunk_size,
            chunk_id=chunk_count
        )

        tokens_left -= current_chunk_size
        chunk_count += 1

    # Signal that generation is complete
    _GENERATION_COMPLETE.set()

    # Wait for all chunks to be processed
    print("Waiting for all chunks to be processed...")
    while True:
        with _CHUNKS_LOCK:
            if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS:
                break
        time.sleep(0.01)

    return carry



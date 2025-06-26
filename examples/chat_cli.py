# chat_cli.py
"""
Streaming Gemma‑JAX CLI chat (optimised).
Run: python chat_cli.py
"""

import os
from pathlib import Path
import sys
import time
import threading
import queue
from functools import partial
from typing import Any, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
#                          0.  Environment setup
# -----------------------------------------------------------------------------
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ROOT_DIR = Path('/content/gemma-jax')
    if not ROOT_DIR.exists():
        raise RuntimeError("❌ Clone gemma‑jax into /content/ before running.")
    os.chdir(ROOT_DIR)
    CKPT_PATH = Path('/content/drive/MyDrive/gemma/4b-it')   # customise
else:
    ROOT_DIR  = Path(__file__).resolve().parent.parent
    CKPT_PATH = ROOT_DIR / '4b'                              # customise

TOKENIZER_PATH = ROOT_DIR / 'tokenizer.model'
assert CKPT_PATH.exists(),  f"Checkpoint {CKPT_PATH} not found"
assert TOKENIZER_PATH.exists(), f"Tokenizer {TOKENIZER_PATH} not found"

# -----------------------------------------------------------------------------
#                          1.  Gemma‑JAX imports
# -----------------------------------------------------------------------------
from gemma_jax.core.cache import (
    KVCache, init_cache, shard_kvcache_with_tree_map, create_cache_partition_spec
)
from gemma_jax.core.inference import greedy_sample         # still optional
from gemma_jax.core.model     import (
    forward_fn, decode # , build_gen_step_attn_masks
)
from gemma_jax.core.rope      import load_rope_cache
from gemma_jax.core.sp_tokenizer import SentencePieceTokenizer, format_prompt
from gemma_jax.core.weights   import (
    create_config as create_gemma_config,
    create_device_mesh, load_model
)
from gemma_jax.core.segment   import SegmentInfo

# ------------------------------------------------------------
#  Shared special‑token ids
# ------------------------------------------------------------
PAD_ID, EOS_ID, BOS_ID, END_OF_TURN_ID = 0, 1, 2, 106
# Tokens that terminate generation early
STOP_TOKENS = (EOS_ID, END_OF_TURN_ID)

# -----------------------------------------------------------------------------
#                          2.  Model / tokenizer
# -----------------------------------------------------------------------------
MODEL_SIZE       = 4
CACHE_LENGTH     = 1024       # 128 # 2_048
CHUNK_LENGTH     = 128        # 32   # 1_024
WINDOW_SIZE      = 1024       # 128 # 2_048
STREAM_PACK      = 8          # <‑‑ number of tokens per host‑callback
DTYPE            = jnp.bfloat16
MAX_TOKENS       = 32         # default generation tokens per turn

mesh    = create_device_mesh()
config  = create_gemma_config(model_size=MODEL_SIZE,
                              batch_size=1,
                              cache_length=CACHE_LENGTH,
                              chunk_length=CHUNK_LENGTH,
                              window_size=WINDOW_SIZE,
                              generate_steps=MAX_TOKENS)

tokenizer  = SentencePieceTokenizer(TOKENIZER_PATH)
rope_cache = None                 # optional; depends on build
model      = load_model(CKPT_PATH, mesh, config, dtype=DTYPE)

# -----------------------------------------------------------------------------
#                          3.  Helper JITs
# -----------------------------------------------------------------------------

def _build_pos(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

@partial(jax.jit, static_argnames=("cfg", "chunk_len", "mesh"))
def _prefill(state, input_ids, offset, model, kv_cache, rope, cfg, *, chunk_len, mesh):
    """
    Fast path: prefill entire prompt in one JIT call
    (we assume single‑turn < CHUNK_LENGTH).
    """
    bsz, seqlen = input_ids.shape
    pad_len     = (chunk_len - seqlen) % chunk_len
    tokens      = jnp.pad(input_ids, ((0,0), (0,pad_len)), constant_values=0)
    attn_mask   = tokens != 0
    pos_ids     = _build_pos(attn_mask) + offset

    write_idx   = jnp.full((bsz,), offset, jnp.int32)
    chunk = cfg.chunk_length       
    q_pos = jnp.arange(chunk, dtype=jnp.int32)[:, None] + offset
    k_pos = jnp.arange(cfg.cache_length, dtype=jnp.int32)[None, :]
    mask_bts = jnp.broadcast_to(k_pos <= q_pos,
                            (bsz, chunk, cfg.cache_length))

    # Build SegmentInfo reflecting state BEFORE this chunk is written
    seg_info = SegmentInfo(
        lengths=jnp.full((bsz,), offset, jnp.int32),   # current seq len before prefill
        cursor=jnp.full((bsz,), offset, jnp.int32),    # write head starts at offset
        offset=jnp.zeros((bsz,), jnp.int32),           # ring buffer origin (0)
        cache_len=int(cfg.cache_length),
    )

    _, kv = forward_fn(
        state,
        tokens,                    # (B,S)
        pos_ids,
        # mask_bts,
        seg_info,
        model=model,
        cache=kv_cache,
        rope_cache=rope,
        config=cfg,
        auto_regressive=False,
        mesh=mesh,
    )
    return kv


@partial(jax.jit, static_argnames=("cfg", "mesh"))
def _gen_one(carry, *, model, rope, cfg, mesh):
    """Generate ONE token using SegmentInfo-aware carry."""
    cur_tok, seg_info, step, kv, state = carry

    B        = cur_tok.shape[0]
    cache_L  = seg_info.cache_len

    # # 1. Causal mask for the current step
    # attn = build_gen_step_attn_masks(
    #     seg_info.cursor[:, None],
    #     cache_L,
    #     jnp.ones((B, cache_L), jnp.bool_),
    # )

    # 2. Forward pass for this token
    x, kv2 = forward_fn(
        state,
        cur_tok[:, None],                  # (B,1)
        seg_info.current_pos[:, None],     # (B,1)
        # attn,
        seg_info,
        model=model,
        cache=kv,
        rope_cache=rope,
        config=cfg,
        auto_regressive=True,
        mesh=mesh,
    )

    logits  = decode(model, x).squeeze(1)
    nxt_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    # 3. Advance SegmentInfo & build new carry
    seg_info2 = seg_info.advance(1)

    carry2 = (
        nxt_tok,
        seg_info2,
        step + 1,
        kv2,
        state,
    )
    return carry2, nxt_tok


@partial(jax.jit, static_argnames=("steps", "pack", "cfg", "mesh"))
def _generate_loop(init_carry, *, steps: int, pack: int, model, rope, cfg, mesh):
    """Generate `steps` tokens in batches of `pack` using SegmentInfo carry."""

    # Inner step that generates ONE token ----------------------------------
    def _gen_one_closure(carry, _):
        return _gen_one(carry, model=model, rope=rope, cfg=cfg, mesh=mesh)

    # Body that produces `pack` tokens -------------------------------------
    def _pack_body(carry, _):
        carry2, tok_pack = jax.lax.scan(_gen_one_closure, carry, xs=None, length=pack)
        return carry2, tok_pack

    num_packs             = steps // pack
    carry_fin, tok_packs  = jax.lax.scan(_pack_body, init_carry, xs=None, length=num_packs)
    return carry_fin, tok_packs.reshape(-1)


# -----------------------------------------------------------------------------
#                          4.  Conversation manager
# -----------------------------------------------------------------------------
class Conversation:
    """
    Stateful single‑stream conversation (batch==1).
    Optimised streaming: emits STREAM_PACK tokens per callback.
    """

    def __init__(self):
        self.mesh  = mesh
        self.state = (jnp.array([42]))        # dummy
        self.kv    = self._init_cache()
        self.seq   = 0                        # tokens so far

    def _init_cache(self):
        cache = init_cache(
            batch            = 1,
            max_seq_len      = config.cache_length,
            num_layers       = config.num_layers,
            num_kv_heads     = config.num_kv_heads,
            head_dim         = config.head_dim,
        )
        mesh_axes = {'batch':'data', 'heads':'model'}
        return shard_kvcache_with_tree_map(cache, self.mesh, mesh_axes)

    # ----------------------- PUBLIC API -----------------------
    def turn(self, prompt:str, gen_tokens:int):
        # 1. Tokenise
        prompt_ids = tokenizer.encode(format_prompt(prompt),
                                      add_bos=True, add_eos=False)
        if self.seq + len(prompt_ids) >= config.cache_length:
            print("⚠️  Context full ─ clearing cache")
            self.clear()
            prompt_ids = tokenizer.encode(format_prompt(prompt),
                                          add_bos=True, add_eos=False)

        # 2. Prefill
        self.kv = _prefill(self.state,
                           jnp.array([prompt_ids], jnp.int32),
                           self.seq, model, self.kv, rope_cache,
                           config, chunk_len=CHUNK_LENGTH, mesh=self.mesh)
        self.seq += len(prompt_ids)
        last_tok  = prompt_ids[-1]

        # 3. Generation carry (SegmentInfo variant)
        seg_info = SegmentInfo(
            lengths=jnp.array([self.seq], jnp.int32),  # tokens already in cache
            cursor=jnp.array([self.seq], jnp.int32),   # next write position
            offset=jnp.array([0],        jnp.int32),   # ring-buffer offset (0 for now)
            cache_len=int(config.cache_length),
        )

        carry0 = (
            jnp.array([last_tok], jnp.int32),  # current token
            seg_info,
            jnp.array(0, jnp.int32),           # step counter
            self.kv,
            self.state,
        )

        # 4.  Autoregressive generation with **early stop**
        carry = carry0
        toks_out: List[int] = []

        while len(toks_out) < gen_tokens:
            # Generate ONE 8‑token pack (pattern from MaxText / JetStream)
            carry, tok_pack = _generate_loop(
                carry, steps=STREAM_PACK, pack=STREAM_PACK,
                model=model, rope=rope_cache, cfg=config, mesh=self.mesh)

            for t in np.asarray(tok_pack).tolist():
                # Token already lives in KV‑cache, so bump the cursor first
                self.seq += 1

                if t in STOP_TOKENS:
                    # Do **not** surface the stop token itself
                    self.kv = carry[-2]
                    return np.array(toks_out, dtype=np.int32)

                toks_out.append(t)
                if len(toks_out) >= gen_tokens:
                    break

        # Exhausted the user budget without seeing a stop token
        self.kv = carry[-2]
        return np.array(toks_out, dtype=np.int32)

    def clear(self):
        self.kv  = self._init_cache()
        self.seq = 0

# -----------------------------------------------------------------------------
#                          5.  Streaming helper thread
# -----------------------------------------------------------------------------
_PRINT_Q: "queue.Queue[List[int]]" = queue.Queue()
_CONSUMER : threading.Thread | None = None
_STOP     = object()

def _consumer():
    print("[decoder started]")
    while True:
        pack = _PRINT_Q.get()
        if pack is _STOP:
            break
        print(tokenizer.decode(pack), end='', flush=True)
        _PRINT_Q.task_done()
    print("\n[decoder stopped]")


def _ensure_consumer():
    global _CONSUMER
    if _CONSUMER is None or not _CONSUMER.is_alive():
        _CONSUMER = threading.Thread(target=_consumer, daemon=True)
        _CONSUMER.start()

def _shutdown_consumer():
    if _CONSUMER and _CONSUMER.is_alive():
        _PRINT_Q.put(_STOP)
        _CONSUMER.join()

# -----------------------------------------------------------------------------
#                          6.  Command‑line UI
# -----------------------------------------------------------------------------
def repl():
    print(f"gemma‑jax {MODEL_SIZE}‑b │ type /exit to quit, /clear to reset cache\n")
    convo  = Conversation()
    gen_tk = MAX_TOKENS # 128
    _ensure_consumer()

    try:
        while True:
            try:
                # Flush stdout to avoid prompt flicker
                sys.stdout.flush()
                user_in = input("You ▸ ").strip()
            except EOFError:
                break                       # Ctrl‑D

            if not user_in:
                continue
            if user_in.startswith("/"):
                cmd, *arg = user_in[1:].split()
                if cmd == "exit": break
                if cmd == "clear":
                    convo.clear()
                    print("✅ cache cleared"); continue
                if cmd == "tokens" and arg:
                    gen_tk = int(arg[0]); print(f"🔧 gen_tokens={gen_tk}"); continue
                print("❓ commands: /exit /clear /tokens N"); continue

            # normal turn ---------------------------------------------------
            print("Gemma ▸ ", end="", flush=True)
            t0 = time.time()
            toks = convo.turn(user_in, gen_tk)         # JAX array
            # stream packs to decoder
            packs = np.array_split(
                np.asarray(toks),
                max(1, (len(toks) + STREAM_PACK - 1) // STREAM_PACK)
            )

            for p in packs: _PRINT_Q.put(p.tolist())
            _PRINT_Q.join()                            # wait until printed
            dt = time.time() - t0
            print(f"\n⏱  {dt:.2f}s,  ({convo.seq} tokens total)")
            print(f"Througput: {convo.seq / dt:.1f} tokens/s", flush=True)

    finally:
        _shutdown_consumer()

# -----------------------------------------------------------------------------
#                          7.  Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    repl()
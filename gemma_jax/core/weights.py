# %% 

"""gemma3_sharding_refactored.py
================================
A JAX-idiomatic helper module for sharding **Gemma-3** checkpoints.

This streamlined version
* keeps *no-boiler-plate* pytree compatibility by using plain
  `typing.NamedTuple` for both **PartitionSpec** trees and runtime model
  states, and
* exposes a tiny public API (`create_config`, `create_device_mesh`,
  `load_model`, `load_params`).

Place the file anywhere on your `$PYTHONPATH`, `pip install orbax` for
checkpoint restore support, and you can immediately shard Gemma-3
checkpoints across CPU, GPU, or TPU meshes.
"""
from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

__all__ = [
    "Config",
    "create_config",
    "create_device_mesh",
    "load_model",
    "load_params",
]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s] %(message)s")
_logger = logging.getLogger("gemma3.shard")

# -----------------------------------------------------------------------------
# Constants & enums
# -----------------------------------------------------------------------------

class SpecialToken(IntEnum):
    PAD = 0
    EOS = 1


class AttentionType(IntEnum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


# Base six-layer attention pattern ------------------------------------------------
_BASE_PATTERN: tuple[AttentionType, ...] = (
    AttentionType.LOCAL_SLIDING,
) * 5 + (AttentionType.GLOBAL,)

# Gemma-3 variant table -----------------------------------------------------------
# fmt: off
_GEMMA3_VARIANTS = np.array([
    #  1B   4B    12B    27B
    [   1,   4,    12,    27],   # model size (B)
    [  26,  34,    48,    62],   # layers
    [   4,   8,    16,    32],   # heads
    [   1,   4,     8,    16],   # kv-heads
    [1152, 2560,  3840,  5376],  # embed-dim
    [6912,10240, 15360, 21504],  # mlp-hidden_dim (6*9*128, 4*20*128, 4*30*128, 4*42*128)
    [ 256,  256,   256,   128],  # head-dim
], dtype=np.int32)
# fmt: on

_ROW = {
    "model_size": 0,
    "num_layers": 1,
    "num_heads": 2,
    "num_kv_heads": 3,
    "embed_dim": 4,
    "hidden_dim": 5,
    "head_dim": 6,
}

# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Config:
    # Architecture
    model_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    hidden_dim: int
    head_dim: int

    # Runtime
    batch_size: int
    window_size: int
    cache_length: int
    chunk_length: int
    # max_gen_length: int
    generate_steps: int

    # Attention
    query_pre_attn_scalar: float
    attention_pattern: tuple[AttentionType, ...] = _BASE_PATTERN

    # Vocab / logits
    vocab_size: int = 262_144
    final_logit_softcap: float | None = None

    # Norm flags
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    use_qk_norm: bool = True

    # Rotary cache
    use_rope_cache: bool = False
    rope_cache: Array | None = None

    # Extra (vision etc.)
    vision_encoder: tuple[int, ...] | None = None
    mm_extra_vocab_size: int = 0

    # Frequencies
    local_base_frequency: int = 10_000
    local_scale_factor: float = 1.0
    global_base_frequency: int = 1_000_000
    global_scale_factor: float = 8.0

    # Misc
    transpose_gating_einsum: bool = True
    attn_logits_soft_cap: float | None = None

    # # Alias for back-compat
    # @property
    # def attention_types(self):
    #     return self.attention_pattern

# -----------------------------------------------------------------------------
# Config factory
# -----------------------------------------------------------------------------

def _tile_pattern(n: int) -> tuple[AttentionType, ...]:
    rep, rem = divmod(n, len(_BASE_PATTERN))
    return _BASE_PATTERN * rep + _BASE_PATTERN[:rem]

def get_query_pre_attn_norm(model_size: int) -> float:
  """Calculate the pre-attention normalization scalar for the query.

  Scales by '1/sqrt(embed_dim // num_heads)' if 27B model,
  otherwise by '1/sqrt(head_dim=256)'.
  """
  head_dim_27b = 128
  embed_dim_27b = 5376
  num_heads_27b = 32
  return (embed_dim_27b / num_heads_27b) ** -0.5 if model_size == 27 else (head_dim_27b * 2) ** -0.5


def create_config(*, model_size: int, batch_size: int, chunk_length: int,
                  window_size: int, cache_length: int, generate_steps: int,
                  **kw) -> Config:
    """Return a fully specified :class:`Config`.  Extra `kw` override defaults."""
    if model_size not in (1, 4, 12, 27):
        raise ValueError("Model size must be 1, 4, 12, or 27 (billions).")

    col = int(np.where(_GEMMA3_VARIANTS[_ROW["model_size"]] == model_size)[0][0])
    base = {k: int(_GEMMA3_VARIANTS[row, col]) for k, row in _ROW.items() if k != "model_size"}

    # q_scalar = ( (base["embed_dim"] / base["num_heads"]) ** -0.5 if model_size == 27 else (256 * 2) ** -0.5) #BUG
    q_scalar = get_query_pre_attn_norm(model_size)

    cfg = Config(
        model_size=model_size,
        batch_size=batch_size,
        chunk_length=chunk_length,
        window_size=window_size,
        cache_length=cache_length,
        # max_gen_length=max_gen_length,
        generate_steps=generate_steps,
        attention_pattern=_tile_pattern(base["num_layers"]),
        query_pre_attn_scalar=q_scalar,
        **base,
    )
    return replace(cfg, **kw)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_T = TypeVar("_T")
FlatDict = dict[str, Array]
NestedDict = dict[str, Any]


def _flatten_pytree_with_dots(tree) -> FlatDict:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)

    def _p2s(p):
        parts: list[str] = []
        for k in p:
            parts.append(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))))
        return ".".join(parts)

    return {_p2s(path): v for path, v in flat}

# -----------------------------------------------------------------------------
# Device mesh creation
# -----------------------------------------------------------------------------

def create_device_mesh(shape: tuple[int | None, int | None] | None = None, *, axis_names=("data", "model")) -> Mesh:
    devs = jax.devices()
    if {d.platform for d in devs} == {"cpu"}:  # trivial 1×1 mesh on CPU
        return Mesh(np.array(devs).reshape((1, 1)), axis_names=axis_names)

    n = len(devs)
    if shape is None or shape == (None, None):  # heuristic square factor
        side = int(math.sqrt(n))
        while side > 1 and n % side:
            side -= 1
        data, model = side, n // side
    else:
        data, model = shape
        if data is None and model is None:
            raise ValueError("shape cannot be (None, None)")
        if data is None:
            if n % model:
                raise ValueError(f"{n} devs can't fit (_, {model})")
            data = n // model
        if model is None:
            if n % data:
                raise ValueError(f"{n} devs can't fit ({data}, _)")
            model = n // data
    if data * model > n:
        raise ValueError("Mesh larger than available devices")

    mesh = Mesh(np.array(devs[: data * model]).reshape((data, model)), axis_names=axis_names)
    _logger.info("Mesh %dx%d constructed on %d devices", data, model, n)
    return mesh

# -----------------------------------------------------------------------------
# Checkpoint loading utils
# -----------------------------------------------------------------------------

@functools.cache
def _restore_ckpt(path: str | Path):
    return ocp.PyTreeCheckpointer().restore(str(path))

def _load_flat_params(path: str | Path, *, dtype: jnp.dtype | None = None, keep_siglip: bool = False) -> FlatDict:
    _logger.info("Reading checkpoint %s", path)
    raw = _restore_ckpt(path)
    flat = _flatten_pytree_with_dots(raw)
    if not keep_siglip:
        flat = {k: v for k, v in flat.items() if not k.startswith("SigLiPFromPatches_0")}
    if dtype:
        flat = jax.tree_util.tree_map(lambda x: x.astype(dtype), flat)
    return {k.replace("/", "."): v for k, v in flat.items()}

# -----------------------------------------------------------------------------
# Gemma-3 runtime structures
# -----------------------------------------------------------------------------

# fmt: off
#  'LayerNorm_0.bias',
#  'LayerNorm_0.scale',
#  'LayerNorm_1.bias',
#  'LayerNorm_1.scale',
#  'MlpBlock_0.Dense_0.bias',
#  'MlpBlock_0.Dense_0.kernel',
#  'MlpBlock_0.Dense_1.bias',
#  'MlpBlock_0.Dense_1.kernel',
#  'MultiHeadDotProductAttention_0.key.bias',
#  'MultiHeadDotProductAttention_0.key.kernel',
#  'MultiHeadDotProductAttention_0.out.bias',
#  'MultiHeadDotProductAttention_0.out.kernel',
#  'MultiHeadDotProductAttention_0.query.bias',
#  'MultiHeadDotProductAttention_0.query.kernel',
#  'MultiHeadDotProductAttention_0.value.bias',
#  'MultiHeadDotProductAttention_0.value.kernel',
#  ]

# SigLiPFromPatches_0.siglip_encoder.pos_embedding                                                               (1, 4096, 1152)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.bias                                               (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.scale                                              (1152,)
# SigLiPFromPatches_0.siglip_encoder.embedding.bias                                                              (1152,)
# SigLiPFromPatches_0.siglip_encoder.embedding.kernel                                                            (14, 14, 3, 1152)

# for SigLiP:
# S = 4096 , E = 1152, M = 4304   (max_seq_len, ViT_embed_dim, mlp_hidden_dim)
# P = 14, C = 3         (patch_size, in_channels)
# vN = 16, vH = 72      (ViT_num_kv_heads, ViT_head_dim)

# fmt: on
class EncoderBlock(NamedTuple):
    layer_norm_0_bias          : Array   # (embed_dim,)                             (E,)
    layer_norm_0_scale         : Array   # (embed_dim,)                             (E,)
    layer_norm_1_bias          : Array   # (embed_dim,)                             (E,)
    layer_norm_1_scale         : Array   # (embed_dim,)                             (E,)

    mlp_block_0_dense_0_bias   : Array   # (mlp_hidden_dim,)                        (M,)                     (4304,)        
    mlp_block_0_dense_0_kernel : Array   # (embed_dim, mlp_hidden_dim)              (E,M)                    (1152, 4304)
    mlp_block_0_dense_1_bias   : Array   # (embed_dim,)                             (E,)                     (1152,)
    mlp_block_0_dense_1_kernel : Array   # (mlp_hidden_dim, embed_dim)              (M,E)                    (4304, 1152)
    key_bias                   : Array   # (num_kv_heads, head_dim)                 (vN,vH)                  (16, 72)
    key_kernel                 : Array   # (embed_dim, num_kv_heads, head_dim)      (E,vN,vH)                (1152, 16, 72)
    out_bias                   : Array   # (embed_dim,)                             (E,)                     (1152,)
    out_kernel                 : Array   # (num_heads, head_dim, embed_dim)         (vN,vH,E)                (16, 72, 1152)
    query_bias                 : Array   # (num_heads, head_dim)                    (vN,vH)                  (16, 72)
    query_kernel               : Array   # (embed_dim, num_heads, head_dim)         (E,vN,vH)                (1152, 16, 72)
    value_bias                 : Array   # (num_heads, head_dim)                    (vN,vH)                  (16, 72)
    value_kernel               : Array   # (embed_dim, num_heads, head_dim)         (E,vN,vH)                (1152, 16, 72)


class Encoder(NamedTuple):
    pos_embedding               : Array   # (1, max_seq_len, embed_dim)             (1,S,E)             
    encoder_norm_scale          : Array   # (embed_dim,)                            (E,)
    encoder_norm_bias           : Array   # (embed_dim,)                            (E,)
    embedding_bias              : Array   # (embed_dim,)                            (E,)
    embedding_kernel            : Array   # (patch_size, patch_size, in_channels, embed_dim)  (P,P,C,E)
    # SigLiP encoder params
    blocks                      : tuple[EncoderBlock, ...]

class Layer(NamedTuple):
    attn_key_norm_scale         : Array    # (head_dim,)                             (H,)
    attn_query_norm_scale       : Array    # (head_dim,)                             (H,)
    output_proj                 : Array    # (num_heads, head_dim, embed_dim)        (N,H,D)
    kv_proj                     : Array    # (2, num_kv_heads, embed_dim, head_dim)  (2,K,D,H)
    q_proj                      : Array    # (num_heads, embed_dim, head_dim)        (N,D,H)
    gating_weights              : Array    # (2, mlp_hidden_dim, embed_dim)          (2,F,D)
    output_weights              : Array    # (mlp_hidden_dim, embed_dim)             (F,D)
    post_attention_norm_scale   : Array    # (embed_dim,)                            (D,)
    post_ffw_norm_scale         : Array    # (embed_dim,)                            (D,)
    pre_attention_norm_scale    : Array    # (embed_dim,)                            (D,)
    pre_ffw_norm_scale          : Array    # (embed_dim,)                            (D,)


class Gemma3(NamedTuple):
    input_embedding_table       : Array     # (vocab_size, embed_dim)                 (V,D)
    mm_input_projection         : Array     # (embed_dim, embed_dim)                  (ViT_embed_dim, D)
    mm_soft_embedding_norm      : Array     # (embed_dim,)                            (ViT_embed_dim,)
    final_norm_scale             : Array     # (embed_dim,)                            (D,)
    blocks                      : tuple[Layer, ...]


class Gemma3MultiModal(NamedTuple):
    input_embedding_table       : Array    # (vocab_size, embed_dim)                  (V,D)
    mm_input_projection         : Array    # (embed_dim, embed_dim)                   (D,D)
    mm_soft_embedding_norm      : Array    # (embed_dim,)                             (D,)
    final_norm_scale             : Array    # (embed_dim,)                             (D,)
    blocks                      : tuple[Layer, ...]
    encoder                     : Encoder
# fmt: off




xs = [
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_0.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_0.scale',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_1.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_1.scale',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_0.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_0.kernel',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_1.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_1.kernel',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.kernel',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.kernel',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.kernel',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.kernel',

 'SigLiPFromPatches_0.siglip_encoder.pos_embedding',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.bias',
 'SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.scale',
 'SigLiPFromPatches_0.siglip_encoder.embedding.bias',
 'SigLiPFromPatches_0.siglip_encoder.embedding.kernel'
]

def _build_shallow_dict_multi_modal(params: FlatDict, *, num_layers: int) -> NestedDict:
    """Return a 2-level dict mirroring the NamedTuple but easy to JSON-dump."""
    encoder_prefix = "SigLiPFromPatches_0.siglip_encoder."
    encoderblock_prefix = "SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_"
    out: NestedDict = {
        "input_embedding_table"         : params["transformer.embedder.input_embedding"],
        "mm_input_projection"           : params["transformer.embedder.mm_input_projection.w"],
        "mm_soft_embedding_norm"        : params["transformer.embedder.mm_soft_embedding_norm.scale"],
        "final_norm_scale"               : params["transformer.final_norm.scale"],
        "blocks"                        : [],
        "encoder": {
            "encoder_norm_scale"           : params[encoder_prefix + "Transformer.encoder_norm.scale"],
            "encoder_norm_bias"            : params[encoder_prefix + "Transformer.encoder_norm.bias"],
            "pos_embedding"                : params[encoder_prefix + "pos_embedding"],
            "embedding_bias"               : params[encoder_prefix + "embedding.bias"],
            "embedding_kernel"             : params[encoder_prefix + "embedding.kernel"],
            "blocks": []
        } 
    }
    for i in range(num_layers):
        p = f"transformer.layer_{i}."
        out["blocks"].append({
            "attn_key_norm_scale"       : params[p + "attn._key_norm.scale"],
            "attn_query_norm_scale"     : params[p + "attn._query_norm.scale"],
            "output_proj"               : params[p + "attn.attn_vec_einsum.w"],
            "kv_proj"                   : params[p + "attn.kv_einsum.w"],
            "q_proj"                    : params[p + "attn.q_einsum.w"],
            "gating_weights"            : params[p + "mlp.gating_einsum.w"],
            "output_weights"            : params[p + "mlp.linear.w"],
            "post_attention_norm_scale" : params[p + "post_attention_norm.scale"],
            "post_ffw_norm_scale"       : params[p + "post_ffw_norm.scale"],
            "pre_attention_norm_scale"  : params[p + "pre_attention_norm.scale"],
            "pre_ffw_norm_scale"        : params[p + "pre_ffw_norm.scale"],
        })

    # Add SigLiP encoder params 
    num_siglip_layers = 27  # Constant for SigLiP
    for i in range(num_siglip_layers):
        p = f"{encoderblock_prefix}{i}."
        out["encoder"]["blocks"].append({
            "LayerNorm_0_bias"         : params[p + "LayerNorm_0.bias"],
            "LayerNorm_0_scale"        : params[p + "LayerNorm_0.scale"],
            "LayerNorm_1_bias"         : params[p + "LayerNorm_1.bias"],
            "LayerNorm_1_scale"        : params[p + "LayerNorm_1.scale"],
            "MlpBlock_0_Dense_0_bias"  : params[p + "MlpBlock_0.Dense_0.bias"],
            "MlpBlock_0_Dense_0_kernel": params[p + "MlpBlock_0.Dense_0.kernel"],
            "MlpBlock_0_Dense_1_bias"  : params[p + "MlpBlock_0.Dense_1.bias"],
            "MlpBlock_0_Dense_1_kernel": params[p + "MlpBlock_0.Dense_1.kernel"],
            "MultiHeadDotProductAttention_0_key_bias": params[p + "MultiHeadDotProductAttention_0.key.bias"],
            "MultiHeadDotProductAttention_0_key_kernel": params[p + "MultiHeadDotProductAttention_0.key.kernel"],
            "MultiHeadDotProductAttention_0_out_bias": params[p + "MultiHeadDotProductAttention_0.out.bias"],
            "MultiHeadDotProductAttention_0_out_kernel": params[p + "MultiHeadDotProductAttention_0.out.kernel"],
            "MultiHeadDotProductAttention_0_query_bias": params[p + "MultiHeadDotProductAttention_0.query.bias"],
            "MultiHeadDotProductAttention_0_query_kernel": params[p + "MultiHeadDotProductAttention_0.query.kernel"],
            "MultiHeadDotProductAttention_0_value_bias": params[p + "MultiHeadDotProductAttention_0.value.bias"],
            "MultiHeadDotProductAttention_0_value_kernel": params[p + "MultiHeadDotProductAttention_0.value.kernel"],
        })

    return out
# fmt: on


# params = _load_flat_params(CHECKPOINT_PATH.as_posix(), dtype=jnp.bfloat16, keep_siglip=True)
# siglip_keys = [k for k in params.keys() if not k.startswith('transformer')]
# for k in xs: print(f"{k:<110} {str(params[k].shape)}")


# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_0.bias                                 (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_0.scale                                (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_1.bias                                 (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.LayerNorm_1.scale                                (1152,)

# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_0.bias                          (4304,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_0.kernel                        (1152, 4304)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_1.bias                          (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MlpBlock_0.Dense_1.kernel                        (4304, 1152)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.bias          (16, 72)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.key.kernel        (1152, 16, 72)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.bias          (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.out.kernel        (16, 72, 1152)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.bias        (16, 72)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.query.kernel      (1152, 16, 72)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.bias        (16, 72)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoderblock_0.MultiHeadDotProductAttention_0.value.kernel      (1152, 16, 72)

# SigLiPFromPatches_0.siglip_encoder.pos_embedding                                                               (1, 4096, 1152)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.bias                                               (1152,)
# SigLiPFromPatches_0.siglip_encoder.Transformer.encoder_norm.scale                                              (1152,)
# SigLiPFromPatches_0.siglip_encoder.embedding.bias                                                              (1152,)
# SigLiPFromPatches_0.siglip_encoder.embedding.kernel                                                            (14, 14, 3, 1152)


# -----------------------------------------------------------------------------
# Param-to-pytree conversions
# -----------------------------------------------------------------------------

def _build_model_pytree_multi_modal(params: FlatDict, *, num_layers: int) -> Gemma3MultiModal:
    """Build a Gemma3MultiModal pytree from a flat parameter dictionary."""
    encoder_prefix = "SigLiPFromPatches_0.siglip_encoder."
    encoderblock_prefix = f"{encoder_prefix}Transformer.encoderblock_"
    num_siglip_layers = 27  # Constant for SigLiP

    def _layer(idx: int) -> Layer:
        p = f"transformer.layer_{idx}."
        return Layer(
            params[p + "attn._key_norm.scale"],
            params[p + "attn._query_norm.scale"],
            params[p + "attn.attn_vec_einsum.w"],
            params[p + "attn.kv_einsum.w"],
            params[p + "attn.q_einsum.w"],
            params[p + "mlp.gating_einsum.w"],
            params[p + "mlp.linear.w"],
            params[p + "post_attention_norm.scale"],
            params[p + "post_ffw_norm.scale"],
            params[p + "pre_attention_norm.scale"],
            params[p + "pre_ffw_norm.scale"],
        )

    def _encoder_layer(idx: int) -> EncoderBlock:
        p = f"{encoderblock_prefix}{idx}."
        attn_p = p + "MultiHeadDotProductAttention_0."
        mlp_p = p + "MlpBlock_0."
        return EncoderBlock(
            layer_norm_0_bias=params[p + "LayerNorm_0.bias"],
            layer_norm_0_scale=params[p + "LayerNorm_0.scale"],
            layer_norm_1_bias=params[p + "LayerNorm_1.bias"],
            layer_norm_1_scale=params[p + "LayerNorm_1.scale"],
            mlp_block_0_dense_0_bias=params[mlp_p + "Dense_0.bias"],
            mlp_block_0_dense_0_kernel=params[mlp_p + "Dense_0.kernel"],
            mlp_block_0_dense_1_bias=params[mlp_p + "Dense_1.bias"],
            mlp_block_0_dense_1_kernel=params[mlp_p + "Dense_1.kernel"],
            key_bias=params[attn_p + "key.bias"],
            key_kernel=params[attn_p + "key.kernel"],
            out_bias=params[attn_p + "out.bias"],
            out_kernel=params[attn_p + "out.kernel"],
            query_bias=params[attn_p + "query.bias"],
            query_kernel=params[attn_p + "query.kernel"],
            value_bias=params[attn_p + "value.bias"],
            value_kernel=params[attn_p + "value.kernel"],
        )

    encoder = Encoder(
        encoder_norm_scale=params[encoder_prefix + "Transformer.encoder_norm.scale"],
        encoder_norm_bias=params[encoder_prefix + "Transformer.encoder_norm.bias"],
        pos_embedding=params[encoder_prefix + "pos_embedding"],
        embedding_bias=params[encoder_prefix + "embedding.bias"],
        embedding_kernel=params[encoder_prefix + "embedding.kernel"],
        blocks=tuple(_encoder_layer(i) for i in range(num_siglip_layers)),
    )

    return Gemma3MultiModal(
        input_embedding_table=params["transformer.embedder.input_embedding"],
        mm_input_projection=params["transformer.embedder.mm_input_projection.w"],
        mm_soft_embedding_norm=params["transformer.embedder.mm_soft_embedding_norm.scale"],
        final_norm_scale=params["transformer.final_norm.scale"],
        blocks=tuple(_layer(i) for i in range(num_layers)),
        encoder=encoder,
    )


def _build_model_pytree(params: FlatDict, *, num_layers: int) -> Gemma3:
    def _layer(idx: int) -> Layer:
        p = f"transformer.layer_{idx}."
        return Layer(
            params[p + "attn._key_norm.scale"],
            params[p + "attn._query_norm.scale"],
            params[p + "attn.attn_vec_einsum.w"],
            params[p + "attn.kv_einsum.w"],
            params[p + "attn.q_einsum.w"],
            params[p + "mlp.gating_einsum.w"],
            params[p + "mlp.linear.w"],
            params[p + "post_attention_norm.scale"],
            params[p + "post_ffw_norm.scale"],
            params[p + "pre_attention_norm.scale"],
            params[p + "pre_ffw_norm.scale"],
        )

    return Gemma3(
        params["transformer.embedder.input_embedding"],
        params["transformer.embedder.mm_input_projection.w"],
        params["transformer.embedder.mm_soft_embedding_norm.scale"],
        params["transformer.final_norm.scale"],
        tuple(_layer(i) for i in range(num_layers)),
    )


# fmt: off
def _build_shallow_dict(params: FlatDict, *, num_layers: int) -> NestedDict:
    """Return a 2-level dict mirroring the NamedTuple but easy to JSON-dump."""
    out: NestedDict = {
        "input_embedding_table"         : params["transformer.embedder.input_embedding"],
        "mm_input_projection"           : params["transformer.embedder.mm_input_projection.w"],
        "mm_soft_embedding_norm"        : params["transformer.embedder.mm_soft_embedding_norm.scale"],
        "final_norm_scale"               : params["transformer.final_norm.scale"],
        "blocks"                        : [],
    }
    for i in range(num_layers):
        p = f"transformer.layer_{i}."
        out["blocks"].append({
            "attn_key_norm_scale"       : params[p + "attn._key_norm.scale"],
            "attn_query_norm_scale"     : params[p + "attn._query_norm.scale"],
            "output_proj"               : params[p + "attn.attn_vec_einsum.w"],
            "kv_proj"                   : params[p + "attn.kv_einsum.w"],
            "q_proj"                    : params[p + "attn.q_einsum.w"],
            "gating_weights"            : params[p + "mlp.gating_einsum.w"],
            "output_weights"            : params[p + "mlp.linear.w"],
            "post_attention_norm_scale" : params[p + "post_attention_norm.scale"],
            "post_ffw_norm_scale"       : params[p + "post_ffw_norm.scale"],
            "pre_attention_norm_scale"  : params[p + "pre_attention_norm.scale"],
            "pre_ffw_norm_scale"        : params[p + "pre_ffw_norm.scale"],
        })
    return out
# fmt: on

# -----------------------------------------------------------------------------
# Sharding wrappers
# -----------------------------------------------------------------------------

def _device_put_with_spec(pytree: _T, spec_tree: _T, mesh: Mesh) -> _T:
    sharding = jax.tree_util.tree_map(lambda s: NamedSharding(mesh, s), spec_tree)
    return jax.device_put(pytree, sharding)

# -----------------------------------------------------------------------------
# Public loading API
# -----------------------------------------------------------------------------

def load_model(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3:
    """Load → build NamedTuple → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree(flat, num_layers=cfg.num_layers)
    spec_pt = _model_spec(cfg)

    _logger.info("Sharding model …")
    t0 = time.perf_counter()
    with mesh:
        sharded = _device_put_with_spec(model_pt, spec_pt, mesh)
        sharded.input_embedding_table.block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return sharded


def old_load_multi_modal_model(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3MultiModal:
    """Load multi-modal checkpoint → build NamedTuple → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype, keep_siglip=True)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree_multi_modal(flat, num_layers=cfg.num_layers)
    # Note: A corresponding _model_spec_multi_modal would be needed for sharding.
    # This example proceeds without sharding the vision model for simplicity.
    # spec_pt = _model_spec_multi_modal(cfg)

    _logger.info("Building multi-modal model...")
    # Example does not include sharding spec for the vision encoder.
    # To shard, you would define a Gemma3MultiModalPSpec and use it here.
    # with mesh:
    #     sharded = _device_put_with_spec(model_pt, spec_pt, mesh)
    #     sharded.input_embedding_table.block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return model_pt


def load_params(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> NestedDict:
    """Load → shallow dict → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    nested = _build_shallow_dict(flat, num_layers=cfg.num_layers)
    pspec = _param_spec_dict(cfg)

    _logger.info("Sharding param-dict …")
    t0 = time.perf_counter()
    with mesh:
        sharded = _device_put_with_spec(nested, pspec, mesh)
        sharded["input_embedding_table"].block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return sharded

def load_unsharded_model(path: str | Path, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3:
    """Load → build NamedTuple → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree(flat, num_layers=cfg.num_layers)
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return model_pt

def load_unsharded_params(path: str | Path, cfg: Config, *, dtype: jnp.dtype | None = None, flatten: bool=False) -> NestedDict:
    """Load → shallow dict  → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    nested = _build_shallow_dict(flat, num_layers=cfg.num_layers)
    if flatten:
        flattened = _flatten_pytree_with_dots(nested)
        _logger.info("Flattened to %d keys", len(flattened))
        _logger.info("Done (%.2fs)", time.perf_counter() - t0)

        return flattened
    _logger.info("Nested dict with %d keys", len(nested))
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return nested

def load_multi_modal_model(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3MultiModal:
    """Load multi-modal checkpoint → build NamedTuple → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype, keep_siglip=True)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree_multi_modal(flat, num_layers=cfg.num_layers)
    spec_pt = _model_spec_multi_modal(cfg)

    _logger.info("Sharding multi-modal model...")
    t0 = time.perf_counter()
    with mesh:
        sharded = _device_put_with_spec(model_pt, spec_pt, mesh)
        sharded.input_embedding_table.block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return sharded

def load_unsharded_multi_modal_model(path: str | Path, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3MultiModal:
    """Load multi-modal checkpoint → build NamedTuple → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype, keep_siglip=True)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree_multi_modal(flat, num_layers=cfg.num_layers)
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return model_pt

# -----------------------------------------------------------------------------
# PartitionSpec templates (NamedTuple PYTree)
# -----------------------------------------------------------------------------

class LayerPSpec(NamedTuple):
    attn_key_norm_scale: P
    attn_query_norm_scale: P
    output_proj: P
    kv_proj: P
    q_proj: P
    gating_weights: P
    output_weights: P
    post_attention_norm_scale: P
    post_ffw_norm_scale: P
    pre_attention_norm_scale: P
    pre_ffw_norm_scale: P

class Gemma3PSpec(NamedTuple):
    input_embedding_table: P
    mm_input_projection: P
    mm_soft_embedding_norm: P
    final_norm_scale: P
    blocks: tuple[LayerPSpec, ...]

class EncoderBlockPSpec(NamedTuple):
    layer_norm_0_bias: P
    layer_norm_0_scale: P
    layer_norm_1_bias: P
    layer_norm_1_scale: P
    mlp_block_0_dense_0_bias: P
    mlp_block_0_dense_0_kernel: P
    mlp_block_0_dense_1_bias: P
    mlp_block_0_dense_1_kernel: P
    key_bias: P
    key_kernel: P
    out_bias: P
    out_kernel: P
    query_bias: P
    query_kernel: P
    value_bias: P
    value_kernel: P


class EncoderPSpec(NamedTuple):
    encoder_norm_scale: P
    encoder_norm_bias: P
    pos_embedding: P
    embedding_bias: P
    embedding_kernel: P
    blocks: tuple[EncoderBlockPSpec, ...]


class Gemma3MultiModalPSpec(NamedTuple):
    input_embedding_table: P
    mm_input_projection: P
    mm_soft_embedding_norm: P
    final_norm_scale: P
    blocks: tuple[LayerPSpec, ...]
    encoder: EncoderPSpec


# lazy constants
_norm = P()
_emb = P(None, "model")


def _layer_spec() -> LayerPSpec:
    return LayerPSpec(
        attn_key_norm_scale=_norm,
        attn_query_norm_scale=_norm,
        output_proj=P(None, None, "model"),
        kv_proj=P(None, None, "model", None),
        q_proj=P(None, "model", None),
        gating_weights=P(None, "model", None),
        output_weights=P("model", None),
        post_attention_norm_scale=_norm,
        post_ffw_norm_scale=_norm,
        pre_attention_norm_scale=_norm,
        pre_ffw_norm_scale=_norm,
    )


def _encoder_block_spec() -> EncoderBlockPSpec:
    return EncoderBlockPSpec(
        layer_norm_0_bias=_norm,
        layer_norm_0_scale=_norm,
        layer_norm_1_bias=_norm,
        layer_norm_1_scale=_norm,
        mlp_block_0_dense_0_bias=_norm,
        mlp_block_0_dense_0_kernel=P(None, "model"),
        mlp_block_0_dense_1_bias=_norm,
        mlp_block_0_dense_1_kernel=P("model", None),
        key_bias=_norm,
        key_kernel=P(None, "model", None),
        out_bias=_norm,
        out_kernel=P(None, None, "model"),
        query_bias=_norm,
        query_kernel=P(None, "model", None),
        value_bias=_norm,
        value_kernel=P(None, "model", None),
    )


def _encoder_spec() -> EncoderPSpec:
    num_siglip_layers = 27  # Constant for SigLiP
    return EncoderPSpec(
        encoder_norm_scale=_norm,
        encoder_norm_bias=_norm,
        pos_embedding=P(None, None, "model"),
        embedding_bias=_norm,
        embedding_kernel=P(None, None, None, "model"),
        blocks=tuple(_encoder_block_spec() for _ in range(num_siglip_layers)),
    )


def _model_spec(cfg: Config) -> Gemma3PSpec:
    return Gemma3PSpec(
        input_embedding_table=_emb,
        mm_input_projection=_emb,
        mm_soft_embedding_norm=_norm,
        final_norm_scale=_norm,
        blocks=tuple(_layer_spec() for _ in range(cfg.num_layers)),
    )


def _model_spec_multi_modal(cfg: Config) -> Gemma3MultiModalPSpec:
    """Return the PartitionSpec pytree for the multi-modal model."""
    gemma3_spec = _model_spec(cfg)
    encoder_spec = _encoder_spec()
    return Gemma3MultiModalPSpec(
        **gemma3_spec._asdict(),
        encoder=encoder_spec,
    )


# -----------------------------------------------------------------------------
# PartitionSpec templates (NamedTuple PYTree)
# -----------------------------------------------------------------------------

# class LayerPSpec(NamedTuple):
#     attn_key_norm_scale: P
#     attn_query_norm_scale: P
#     output_proj: P
#     kv_proj: P
#     q_proj: P
#     gating_weights: P
#     output_weights: P
#     post_attention_norm_scale: P
#     post_ffw_norm_scale: P
#     pre_attention_norm_scale: P
#     pre_ffw_norm_scale: P


# class Gemma3PSpec(NamedTuple):
#     input_embedding_table: P
#     mm_input_projection: P
#     mm_soft_embedding_norm: P
#     final_norm_scale: P
#     blocks: tuple[LayerPSpec, ...]


# # lazy constants
# _norm = P()
# _emb = P(None, "model")


# def _layer_spec() -> Layer:
#     return Layer(
#         _norm,
#         _norm,
#         P(None, None, "model"),
#         P(None, None, "model", None),
#         P(None, "model", None),
#         P(None, "model", None),
#         P("model", None),
#         _norm,
#         _norm,
#         _norm,
#         _norm,
#     )


# def _model_spec(cfg: Config) -> Gemma3:
#     return Gemma3(_emb, _emb, _norm, _norm, tuple(_layer_spec() for _ in range(cfg.num_layers)))


# -----------------------------------------------------------------------------
# Param-spec as dict (for load_params)
# -----------------------------------------------------------------------------

# fmt: off
def _param_spec_dict(cfg: Config) -> NestedDict:
    norm, emb = _norm, _emb
    layer = {
        "attn_key_norm_scale"       : norm,
        "attn_query_norm_scale"     : norm,
        "output_proj"               : P(None, None, "model"),
        "kv_proj"                   : P(None, None, "model", None),
        "q_proj"                    : P(None, "model", None),
        "gating_weights"            : P(None, "model", None),
        "output_weights"            : P("model", None),
        "post_attention_norm_scale" : norm,
        "post_ffw_norm_scale"       : norm,
        "pre_attention_norm_scale"  : norm,
        "pre_ffw_norm_scale"        : norm,
    }
    return {
        "input_embedding_table"     : emb,
        "mm_input_projection"       : emb,
        "mm_soft_embedding_norm"    : norm,
        "final_norm_scale"           : norm,
        "blocks"                    : [layer] * cfg.num_layers,
    }
# fmt: on
# %%

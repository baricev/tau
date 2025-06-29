# Gemma 3 Multimodal Implementation Files

This document lists the key files that implement and support Gemma 3's multimodal functionality, including image processing, vision encoding, and integration of image tokens with text tokens.

## Core Multimodal Implementation Files

### 1. **Primary Multimodal Module** (`gemma/multimodal/`)
- **`gemma/multimodal/vision.py`** - SigLiP vision encoder implementation
- **`gemma/multimodal/vision_utils.py`** - Vision utilities and helper functions
- **`gemma/multimodal/image.py`** - Image preprocessing and handling

### 2. **Vision Token Processing** (`gemma/gm/vision/`)
- **`gemma/gm/vision/_token_utils.py`** - Token merging and image token integration
- **`gemma/gm/vision/_token_utils_test.py`** - Tests for token utilities
- **`gemma/gm/vision/_preprocess.py`** - Vision preprocessing utilities

### 3. **Core Model Integration** (`gemma/gm/nn/`)
- **`gemma/gm/nn/_transformer.py`** - Main transformer with multimodal support
- **`gemma/gm/nn/_modules.py`** - Embedder with vision projection layer
- **`gemma/gm/nn/_gemma.py`** - Gemma3 model definitions with multimodal configs
- **`gemma/gm/nn/_config.py`** - Configuration classes for multimodal models

### 4. **Inference Integration** (`gemma/gm/text/`)
- **`gemma/gm/text/_sampler.py`** - Main sampler with image input support
- **`gemma/gm/text/_chat_sampler.py`** - Chat interface with multimodal support
- **`gemma/gm/text/_prefill.py`** - Prefill mechanism for text+image sequences
- **`gemma/gm/text/_sampler_loop.py`** - Core sampling loop with multimodal tokens

### 5. **Supporting Infrastructure**
- **`gemma/gm/utils/_attention_mask.py`** - Attention mask computation for mixed modalities
- **`gemma/gm/utils/_cache_helper.py`** - KV cache management for multimodal sequences
- **`gemma/gm/utils/_types.py`** - Type definitions for multimodal inputs
- **`gemma/gm/data/_functional.py`** - Data processing for multimodal tasks
- **`gemma/gm/ckpts/_checkpoint.py`** - Checkpoint handling for multimodal models

### 6. **Examples and Documentation**
- **`examples/multimodal.py`** - Complete multimodal training example
- **`colabs/multimodal.ipynb`** - Interactive notebook demonstrating usage
- **`docs/multimodal.md`** - Documentation for multimodal features

## Key Functionality by File

### Image Processing & Vision Encoding
- **`gemma/multimodal/image.py`** - JPEG encoding, resizing (896×896), normalization
- **`gemma/multimodal/vision.py`** - SigLiP encoder (4096→256 token compression)
- **`gemma/multimodal/vision_utils.py`** - Vision Transformer utilities and configurations

### Token Integration & Merging
- **`gemma/gm/vision/_token_utils.py`** - Merges vision tokens with text tokens at specific positions
- **`gemma/gm/nn/_modules.py`** - Embedder projects vision features (1152D) to text space (2560D)
- **`gemma/gm/vision/_preprocess.py`** - Preprocessing utilities for vision inputs

### Model Architecture Integration
- **`gemma/gm/nn/_transformer.py`** - Orchestrates multimodal forward pass, handles vision encoding
- **`gemma/gm/nn/_gemma.py`** - Defines Gemma3 variants with vision encoder configurations
- **`gemma/gm/nn/_config.py`** - Configuration dataclasses for multimodal model parameters

### Inference Pipeline
- **`gemma/gm/text/_sampler.py`** - Handles image inputs during text generation
- **`gemma/gm/text/_chat_sampler.py`** - Chat interface supporting image inputs
- **`gemma/gm/text/_prefill.py`** - Manages mixed text/image sequences during prefill
- **`gemma/gm/text/_sampler_loop.py`** - Core autoregressive loop with multimodal token support

### Supporting Infrastructure
- **`gemma/gm/utils/_attention_mask.py`** - Computes attention masks for text+vision sequences
- **`gemma/gm/utils/_cache_helper.py`** - KV cache management for variable-length multimodal sequences
- **`gemma/gm/utils/_types.py`** - Type definitions for images, multimodal inputs, and outputs
- **`gemma/gm/data/_functional.py`** - Data transformation functions for multimodal training
- **`gemma/gm/ckpts/_checkpoint.py`** - Loading/saving multimodal model checkpoints

## Architecture Overview

The multimodal functionality works through **token-level fusion**, where:

1. **Images** are processed through SigLiP vision encoder into 256 soft tokens
2. **Text tokens** include special `<start_of_image>` placeholders that expand to accommodate vision tokens
3. **Vision and text tokens** are merged into a unified sequence processed by the same transformer
4. **Attention mechanisms** handle both modalities seamlessly with proper masking

## Key Features

- **SigLiP Integration**: 27-layer Vision Transformer with 1152 hidden dimensions
- **Token Compression**: 4096→256 token reduction per image (93.75% compression)
- **Unified Processing**: Same transformer architecture for both text and vision
- **Efficient Memory**: Optimized token merging and cache management
- **Production Ready**: Full JAX optimization with JIT compilation support

## Usage Entry Points

- **High-level**: `ChatSampler` with image inputs for conversational use
- **Mid-level**: `Sampler` class for batch processing with images
- **Low-level**: Direct `model.apply()` calls with multimodal inputs
- **Training**: Use `examples/multimodal.py` as starting point for custom training

This architecture enables seamless vision-language processing while maintaining the efficiency and performance characteristics of the base Gemma text models.
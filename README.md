# Gemma JAX Research Codebase

A functional, JAX only, implementation of Google's Gemma3 Transformer model, specifically tailored for rapid prototyping and experimentation.

Designed to facilitate easy generation of synthetic datasets and GRPO training.

It emphasizes functional programming paradigms, minimal dependencies, and simple, modular code that's easy to hack.


## Structure

- `gemma_jax/`: The main package source code.
  - `core/`: Contains the core model, training, and inference logic.
- `examples/`: Contains example scripts demonstrating usage.

## Setup

```bash
# Install the package in editable mode
pip install -e .
```

## Repository Contents

- **Gemma 3 Model:**
  - Full, text only, implementation of 1B, 4B, 12B and 27B Gemma-3 models.
  Checkpoint loading and sharding code in `weights.py`.

- **Multi-Turn, Batched Inference Engine:**
  - A scalable "engine" designed for batched conversational scenarios: allows for various prompt configurations, tokenization schemes, and decoding strategies.

- **Flexible Transformer Architecture:**
  - Modular codebase that makes it straightforward to experiment with alternative transformer components, positional encodings, attention mechanisms, and layer normalization techniques.

- **Evaluation:**
  - Example code for 8-shot and 0-shot and COT evaluation using the GSM8K dataset.

- **RL Pipeline:**
  - TODO

## Getting Started

Follow the official documentation to download either a Kaggle or HuggingFace checkpoint. Install JAX for TPU and a few other requirements:

```bash
git clone https://github.com/yourusername/gemma-jax.git
cd gemma-jax
pip install -e .
```

### Example Usage

Run a batched inference example as a notebook or script:

```bash
python examples/multi_turn_chat.py
```

Benchmark with GSM8K (8-shot):

```bash
python examples/gsm8k.py
```

Generate multi-turn synthetic data for RL training:

```bash
python examples/grpo.py
```

Basic chat interface CLI:

```bash
python examples/chat_cli.py
```

### Running Tests

To run the test files directly with `python gemma_jax/tests/test_*.py`, first set
the repository root on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python gemma_jax/tests/test_fused_attention.py  # for example
```

## Acknowledgements

* Special thanks to Google's TPU Research Cloud for their generous support. Please see [Google TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) for more information on the program.

* Model code based on Google DeepMind's implementation:  [google-deepmind/gemma](https://github.com/google-deepmind/gemma).

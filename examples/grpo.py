# %%
# synthetic_reasoning_and_tools.py
import json
import time
from functools import partial
from typing import List, Any, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import threading
import queue
import re

# Assume setup.py provides these correctly configured components
from setup import (
    model,
    tokenizer,
    config,
    state,
    cache,
    rope_cache,
    encode_text,
    reset_threading_state,
)

from gemma_jax.core.model import forward_fn, decode
from gemma_jax.core.segment import SegmentInfo
from gemma_jax.core.cache import KVCache

# --- START: New Components for Reasoning and Tool Use ---

# 1. Define Special Tokens and Tool Schema
# These tags will be used in prompts and parsed from model output.
THINKING_TAG_START = "<thinking>"
THINKING_TAG_END = "</thinking>"
FUNCTION_CALL_TAG_START = "<function_call>"
FUNCTION_CALL_TAG_END = "</function_call>"

# Define a simple schema for available tools.
# In a real system, this could be loaded from a config file.
AVAILABLE_TOOLS = {
    "calculator": {
        "description": "A simple calculator for arithmetic operations. Use it for any math calculation.",
        "usage": "calculator(expression: str)",
    },
    "web_search": {
        "description": "A web search engine. Use it to find up-to-date information or facts about topics, people, and events.",
        "usage": "web_search(query: str)",
    }
}

# 2. Tool Executor Simulation
# This function simulates the execution of a tool call.
def tool_executor(function_call_str: str) -> str:
    """Parses and 'executes' a function call, returning a simulated result."""
    print(f"[Tool Executor] Received call: {function_call_str}")
    if function_call_str.startswith("calculator"):
        try:
            # A more robust parser would be needed for real use.
            expr = function_call_str.replace("calculator(", "")[:-1]
            result = eval(expr, {"__builtins__": {}}, {}) # Safe eval
            return f"The result of the calculation is {result}."
        except Exception as e:
            return f"Error executing calculation: {e}"
    elif function_call_str.startswith("web_search"):
        query = function_call_str.replace("web_search(", "")[:-1].strip("'\"")
        # In a real system, you'd call a search API. Here we simulate it.
        if "president" in query.lower():
            return "According to a web search, the current US President is Joe Biden."
        else:
            return "According to a web search, information on that topic is available online."
    else:
        return "Unknown function call."

# 3. Advanced Metaprompt Generation
def create_instruction_batch(batch_size: int, rng_seed: int) -> List[str]:
    """Generates a batch of sophisticated metaprompts to guide the teacher model."""
    rng = np.random.RandomState(rng_seed)
    metaprompts = []

    # Define metaprompt templates
    reasoning_template = """You are an AI data generator. Create a complex question that requires step-by-step reasoning.
First, provide your thinking process inside <thinking> tags. Then, provide the final answer.
Question: {question}
Assistant:"""

    tool_use_template = """You are an AI data generator that can use tools. You have access to the following tools:
{tools_description}
To use a tool, output a <function_call> tag with the exact function call.
Based on the question below, decide if a tool is needed. If so, call it. Otherwise, answer directly.
Question: {question}
Assistant:"""

    # Sample questions designed to trigger reasoning or tool use
    reasoning_questions = [
        "If a train leaves City A at 8 AM traveling at 60 mph and a car leaves City B (300 miles away) at 9 AM traveling towards City A at 70 mph, at what time will they meet?",
        "Sarah is twice as old as her brother, Tom. In 5 years, she will be 1.5 times as old as him. How old are they now?",
    ]
    tool_use_questions = [
        "What is (14 * 3) + 98 / 2?",
        "Who is the current president of the United States?",
    ]

    for _ in range(batch_size):
        task_type = rng.choice(["reasoning", "tool_use"])
        if task_type == "reasoning":
            question = rng.choice(reasoning_questions)
            metaprompts.append(reasoning_template.format(question=question))
        else: # tool_use
            question = rng.choice(tool_use_questions)
            tools_desc = "\n".join([f"- {tool}: {details['description']} Usage: {details['usage']}" for tool, details in AVAILABLE_TOOLS.items()])
            metaprompts.append(tool_use_template.format(question=question, tools_description=tools_desc))

    return metaprompts

# --- END: New Components ---

# --- START: Modified and Existing Code ---

# The consumer and JAX functions remain largely the same, as they operate on token IDs.
# The core changes are in the Python driver logic.
_CHUNK_QUEUE = queue.Queue()
_GENERATION_COMPLETE = threading.Event()
_CONSUMER_STARTED = threading.Event()
_EXPECTED_CHUNKS = 0
_PROCESSED_CHUNKS = 0
_CHUNKS_LOCK = threading.Lock()
_consumer_t = None
_COLLECTED_TOKENS = {}
_TOKENS_LOCK = threading.Lock()

PAD_ID, EOS_ID, BOS_ID, END_OF_TURN_ID = 0, 1, 2, 106

def consumer_thread_with_collection():
    global _PROCESSED_CHUNKS, _COLLECTED_TOKENS
    _CONSUMER_STARTED.set()
    with _TOKENS_LOCK:
        _COLLECTED_TOKENS.clear()
    while True:
        try: item = _CHUNK_QUEUE.get(timeout=0.1)
        except:
            if _GENERATION_COMPLETE.is_set():
                with _CHUNKS_LOCK:
                    if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS: break
            continue
        if item is None: break
        with _TOKENS_LOCK: _COLLECTED_TOKENS[item["chunk_id"]] = item["chunk"]
        with _CHUNKS_LOCK: _PROCESSED_CHUNKS += 1
        _CHUNK_QUEUE.task_done()
    print("[Consumer thread] Exiting.")

def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

@jax.jit
def setup_carry(*, state: Any, input_ids: jax.Array, prefill_cache: Any) -> tuple:
    B = input_ids.shape[0]
    seq_lens = (input_ids != 0).sum(axis=-1)
    last_tok = input_ids[jnp.arange(B), seq_lens - 1]
    seg_info = SegmentInfo(lengths=seq_lens, cursor=seq_lens, offset=jnp.zeros_like(seq_lens), cache_len=int(prefill_cache.max_seq_len))
    return (last_tok, seg_info, 0, prefill_cache, state)

@partial(jax.jit, static_argnames=("config", "chunk_length", "cache_length"))
def chunked_prefill(state: Any, input_ids: jax.Array, model: Any, cache: KVCache, rope_cache: Any, config: Any, *, chunk_length: int, cache_length: int) -> KVCache:
    batch_size, seq_len = input_ids.shape
    num_chunks = seq_len // chunk_length + (seq_len % chunk_length > 0)
    pad_len = num_chunks * chunk_length - seq_len
    padded_input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)
    padded_attn_mask = padded_input_ids != 0
    padded_position_ids = build_positions(padded_attn_mask)
    def body(carry, idx):
        kv_cache, model_state = carry
        start = idx * chunk_length
        tok_chunk = jax.lax.dynamic_slice(padded_input_ids, (0, start), (batch_size, chunk_length))
        pos_chunk = jax.lax.dynamic_slice(padded_position_ids, (0, start), (batch_size, chunk_length))
        write_idx_B = jnp.full((batch_size,), start, dtype=jnp.int32)
        seg_info = SegmentInfo(lengths=write_idx_B, cursor=write_idx_B, offset=jnp.zeros_like(write_idx_B), cache_len=int(cache_length))
        _, updated_cache = forward_fn(model_state, tok_chunk, pos_chunk, seg_info, model=model, cache=kv_cache, rope_cache=rope_cache, config=config, auto_regressive=False, mesh=None)
        return (updated_cache, model_state), None
    (filled_cache, _), _ = jax.lax.scan(body, (cache, state), jnp.arange(num_chunks))
    return filled_cache

@partial(jax.jit, static_argnames=("config"))
def _generate_one_step(carry, *, model, rope_cache, config):
    (cur_tok, seg_info, step, kv_cache, model_state) = carry
    x_emb, updated_kv_cache = forward_fn(model_state, cur_tok[:, None], seg_info.current_pos[:, None], seg_info, model=model, cache=kv_cache, rope_cache=rope_cache, config=config, auto_regressive=True, mesh=None)
    logits = decode(model, x_emb).squeeze(axis=1)
    next_tok = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    new_carry = (next_tok, seg_info.advance(1), step + 1, updated_kv_cache, model_state)
    return new_carry, next_tok

@partial(jax.jit, static_argnames=("config", "chunk_size", "chunk_id"))
def paxml_generate_chunked_scan_queue(init_carry, *, model, rope_cache, config, chunk_size: int, chunk_id: int):
    def step_fn(carry, _):
        new_carry, next_tok = _generate_one_step(carry, model=model, rope_cache=rope_cache, config=config)
        return new_carry, next_tok
    final_carry, tokens_chunk = jax.lax.scan(step_fn, init_carry, xs=None, length=chunk_size)
    def tap_chunk(dev_chunk):
        chunk_host = jax.device_get(dev_chunk)
        _CHUNK_QUEUE.put({"chunk_id": chunk_id, "chunk": chunk_host})
    jax.experimental.io_callback(tap_chunk, None, tokens_chunk, ordered=True)
    return final_carry, tokens_chunk

def run_generation_and_collect_tokens(init_carry, *, model, rope_cache, config, total_tokens: int, chunk_size: int):
    global _consumer_t, _EXPECTED_CHUNKS, _PROCESSED_CHUNKS, _COLLECTED_TOKENS
    with _CHUNKS_LOCK:
        _PROCESSED_CHUNKS = 0
        _EXPECTED_CHUNKS = (total_tokens + chunk_size - 1) // chunk_size
    _GENERATION_COMPLETE.clear(); _CONSUMER_STARTED.clear()
    if _consumer_t is None or not _consumer_t.is_alive():
        _consumer_t = threading.Thread(target=consumer_thread_with_collection)
        _consumer_t.start()
        _CONSUMER_STARTED.wait()
    carry = init_carry; tokens_left = total_tokens; chunk_count = 0
    while tokens_left > 0:
        current_chunk_size = min(tokens_left, chunk_size)
        carry, _ = paxml_generate_chunked_scan_queue(carry, model=model, rope_cache=rope_cache, config=config, chunk_size=current_chunk_size, chunk_id=chunk_count)
        tokens_left -= current_chunk_size
        chunk_count += 1
    _GENERATION_COMPLETE.set()
    print("Waiting for all chunks to be processed...")
    while True:
        with _CHUNKS_LOCK:
            if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS: break
        time.sleep(0.01)
    all_tokens = []
    with _TOKENS_LOCK:
        for i in range(chunk_count):
            if i in _COLLECTED_TOKENS: all_tokens.append(_COLLECTED_TOKENS[i])
    if all_tokens:
        concatenated = np.concatenate(all_tokens, axis=0)
        return concatenated
    else:
        return None

# MODIFIED: The main generation loop is now more complex to handle tool use.
def run_reasoning_generation(
    model, tokenizer, config, state, cache, rope_cache,
    num_conversations: int = config.batch_size, # 8,
    max_tokens_per_turn: int = config.generate_steps, # 512, # Increased for reasoning
    output_path: str = "reasoning_conversations.jsonl"
):
    """Main driver for generating complex reasoning and tool-use data."""
    cache_length = config.cache_length
    chunk_length = config.chunk_length

    # Store full conversation history as list of message dicts
    conversations_history = [[{"role": "system", "content": "You are a helpful AI assistant."}] for _ in range(num_conversations)]

    print(f"\nGenerating {num_conversations} complex conversations...")

    # Generate the initial metaprompts
    initial_prompts = create_instruction_batch(num_conversations, rng_seed=42)

    # Prepare prompts for the model
    # The full conversation history is formatted for the model at each step.
    prompts_for_model = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) + f"\nUser: {initial_prompts[i]}\nAssistant:"
        for i, conv in enumerate(conversations_history)
    ]

    for i in range(num_conversations):
        conversations_history[i].append({"role": "user", "content": initial_prompts[i]})

    while True: # Loop until all conversations are complete
        # Encode and prefill
        input_ids = encode_text(prompts_for_model, tokenizer, add_special_tokens=False, return_tensors="np")
        prefill_cache = chunked_prefill(
            state, input_ids, model, cache, rope_cache, config,
            chunk_length=chunk_length, cache_length=cache_length
        )
        init_carry = setup_carry(state=state, input_ids=input_ids, prefill_cache=prefill_cache)

        # Generate the next turn from the model
        generated_tokens = run_generation_and_collect_tokens(
            init_carry, model=model, rope_cache=rope_cache, config=config,
            total_tokens=max_tokens_per_turn, chunk_size=chunk_length
        )

        if generated_tokens is None:
            print("Generation finished for all conversations.")
            break

        needs_another_turn = [False] * num_conversations
        next_prompts_for_model = [""] * num_conversations

        # Process the output for each conversation in the batch
        for i in range(num_conversations):
            tokens = [int(t) for t in generated_tokens[:, i] if t not in [PAD_ID, EOS_ID, END_OF_TURN_ID]]
            response = tokenizer.decode(tokens).strip()

            # The core logic for handling different response types
            if FUNCTION_CALL_TAG_START in response:
                # 1. It's a tool call
                match = re.search(f"{re.escape(FUNCTION_CALL_TAG_START)}(.*?){re.escape(FUNCTION_CALL_TAG_END)}", response, re.DOTALL)
                function_call = match.group(1).strip()
                tool_output = tool_executor(function_call)

                # Append the tool call and tool output to the history
                conversations_history[i].append({"role": "assistant", "content": f"{FUNCTION_CALL_TAG_START}{function_call}{FUNCTION_CALL_TAG_END}"})
                conversations_history[i].append({"role": "tool", "content": tool_output})

                # We need to call the model again with the tool's result
                needs_another_turn[i] = True
                next_prompts_for_model[i] = tokenizer.apply_chat_template(conversations_history[i], tokenize=False, add_generation_prompt=True)

            else:
                # 2. It's a standard response (potentially with thinking)
                conversations_history[i].append({"role": "assistant", "content": response})
                # This conversation is done for this "macro" turn.
                needs_another_turn[i] = False

        # If no conversation needs another turn, we are done
        if not any(needs_another_turn):
            break
        else:
            # Otherwise, update the prompts for the conversations that need to continue
            prompts_for_model = next_prompts_for_model
            print("Some conversations require a tool-use follow-up. Continuing...")

    # Save the completed, rich conversations
    with open(output_path, 'w') as f:
        for conv_history in conversations_history:
            # We remove the initial system prompt for cleaner data, but it can be kept.
            final_data = {"messages": conv_history[1:]}
            f.write(json.dumps(final_data) + '\n')

    print(f"\nSaved {num_conversations} complex conversations to {output_path}")
    return conversations_history


# --- START: Enhanced RL / Advanced Reasoning Components ---

# Function to generate multiple outputs per input
# Useful for reinforcement learning, sampling, or reasoning analysis.
def generate_multiple_outputs(
    input_prompt: str,
    model, tokenizer, config, state, cache, rope_cache,
    num_outputs: int = 5,
    max_tokens_per_output: int = 256
) -> List[Dict[str, Any]]:
    """Generates multiple outputs for a single input prompt."""
    outputs = []
    input_ids = encode_text(
        [input_prompt] * num_outputs,
        tokenizer,
        add_special_tokens=False,
        return_tensors="np"
    )

    prefill_cache = chunked_prefill(
        state, input_ids, model, cache, rope_cache, config,
        chunk_length=config.chunk_length,
        cache_length=config.cache_length
    )

    init_carry = setup_carry(
        state=state,
        input_ids=input_ids,
        prefill_cache=prefill_cache
    )

    generated_tokens = run_generation_and_collect_tokens(
        init_carry,
        model=model,
        rope_cache=rope_cache,
        config=config,
        total_tokens=max_tokens_per_output,
        chunk_size=config.chunk_length
    )

    if generated_tokens is None:
        return []

    for i in range(num_outputs):
        tokens = [int(t) for t in generated_tokens[:, i] if t not in [PAD_ID, EOS_ID, END_OF_TURN_ID]]
        response = tokenizer.decode(tokens).strip()
        outputs.append({
            "output_index": i,
            "response": response
        })

    return outputs

# Example reinforcement-learning-like usage
# Generate multiple outputs and select or evaluate them based on custom criteria

def evaluate_outputs(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates multiple generated outputs and returns the best one based on custom criteria."""
    # Placeholder evaluation: here we use simple length-based heuristic
    best_output = max(outputs, key=lambda x: len(x['response']))
    return best_output

# --- Example of Using Advanced Generation ---

def run_advanced_reasoning_generation(
    model, tokenizer, config, state, cache, rope_cache,
    input_prompts: List[str],
    num_outputs_per_input: int = 5,
    max_tokens_per_output: int = config.generate_steps
):
    all_results = []

    for prompt in input_prompts:
        generated_outputs = generate_multiple_outputs(
            prompt,
            model,
            tokenizer,
            config,
            state,
            cache,
            rope_cache,
            num_outputs=num_outputs_per_input,
            max_tokens_per_output=max_tokens_per_output
        )

        best_output = evaluate_outputs(generated_outputs)
        reasoning_trace = {
            "prompt": prompt,
            "generated_outputs": generated_outputs,
            "selected_output": best_output
        }

        all_results.append(reasoning_trace)
        print(f"Prompt: {prompt}\nSelected Output: {best_output['response']}\n")

    # Save results
    with open("advanced_reasoning_traces.jsonl", "w") as f:
        for trace in all_results:
            f.write(json.dumps(trace) + '\n')

    return all_results

# --- Example execution ---
if __name__ == "__main__":
    reset_threading_state()

    input_prompts = [
        "Explain how gravity affects ocean tides in detail.",
        "Calculate the volume of a cylinder with radius 5 and height 10."
    ]

    advanced_traces = run_advanced_reasoning_generation(
        model=model,
        tokenizer=tokenizer,
        config=config,
        state=state,
        cache=cache,
        rope_cache=rope_cache,
        input_prompts=input_prompts,
        num_outputs_per_input=config.batch_size,
        max_tokens_per_output=config.generate_steps
    )

    if _consumer_t and _consumer_t.is_alive():
        _CHUNK_QUEUE.put(None)
        _consumer_t.join(timeout=1.0)

    # Print the first advanced reasoning trace
    print("\n--- Example Advanced Reasoning Trace ---")
    print(json.dumps(advanced_traces[0], indent=2))


'''
if __name__ == "__main__":
    reset_threading_state()

    final_conversations = run_reasoning_generation(
        model=model, tokenizer=tokenizer, config=config, state=state,
        cache=cache, rope_cache=rope_cache,
        num_conversations=config.batch_size,
        max_tokens_per_turn=config.generate_steps,
        output_path="synthetic_reasoning_conversations.jsonl"
    )

    if _consumer_t and _consumer_t.is_alive():
        _CHUNK_QUEUE.put(None)
        _consumer_t.join(timeout=1.0)

    print("\n--- Example Generated Conversation ---")
    if final_conversations:
        # Using a nice format for printing the first conversation
        print(json.dumps(final_conversations[0], indent=2))

'''

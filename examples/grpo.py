# %%
# synthetic_conversations.py
import json
import time
from functools import partial
from typing import List, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import threading

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
    _CHUNK_QUEUE,
    _consumer_t,
    _GENERATION_COMPLETE,
    _CONSUMER_STARTED,
    _EXPECTED_CHUNKS,
    _PROCESSED_CHUNKS,
    _CHUNKS_LOCK
)

PAD_ID, EOS_ID, BOS_ID, END_OF_TURN_ID = 0, 1, 2, 106

CONVERSATION_SEEDS = [
    "Explain quantum computing to a 10-year-old",
    "Help me debug a Python function that's running slowly",
    "What are the ethical implications of AI in healthcare?",
    "Teach me how to make sourdough bread from scratch",
    "Explain the 2008 financial crisis in simple terms",
    "How do I optimize my code for better performance?",
    "What's the difference between machine learning and deep learning?",
    "Help me plan a sustainable garden for my backyard"
]

USER_TURN_TEMPLATES = [
    "Can you explain more about {}?",
    "What would happen if {}?",
    "How does this relate to {}?",
    "Can you give me an example of {}?",
    "What are the main challenges with {}?",
    "How would you implement {}?",
    "What are the alternatives to {}?",
    "Can you compare this with {}?"
]

# Global storage for collected tokens
_COLLECTED_TOKENS = {}
_TOKENS_LOCK = threading.Lock()

def consumer_thread_with_collection():
    """Modified consumer thread that collects tokens instead of just printing."""
    global _PROCESSED_CHUNKS, _COLLECTED_TOKENS
    _CONSUMER_STARTED.set()
    
    # Reset collected tokens for this generation
    with _TOKENS_LOCK:
        _COLLECTED_TOKENS.clear()
    
    while True:
        try:
            item = _CHUNK_QUEUE.get(timeout=0.1)
        except:
            if _GENERATION_COMPLETE.is_set():
                with _CHUNKS_LOCK:
                    if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS:
                        break
            continue
            
        if item is None:  # Poison pill
            break
            
        chunk = item["chunk"]  # shape (chunk_size, B)
        chunk_id = item["chunk_id"]
        
        # Store the raw tokens
        with _TOKENS_LOCK:
            _COLLECTED_TOKENS[chunk_id] = chunk
        
        with _CHUNKS_LOCK:
            _PROCESSED_CHUNKS += 1
        
        _CHUNK_QUEUE.task_done()
    
    print("[Consumer thread] Exiting.")

def run_generation_and_collect_tokens(
    init_carry,
    *,
    model,
    rope_cache,
    config,
    total_tokens: int,
    chunk_size: int
):
    """
    Python driver that calls the chunked scan function (paxml_generate_chunked_scan_queue),repeatedly until 
    we've generated `total_tokens`,  and collects thems to a list

    Under the hood, it uses a consumer thread to process chunks
    and collect tokens in order. This allows us to handle large 
    token generation tasks without running out of memory or
    blocking the main thread.

    paxml_generate_chunked_scan_queue is a single compiled function that does `chunk_size` steps in one scan,
    then returns the entire chunk. Minimizes callback overhead:
    - Just device->host copy
    - Then put data on a queue for async decode
 
    """
    global _consumer_t, _EXPECTED_CHUNKS, _PROCESSED_CHUNKS, _COLLECTED_TOKENS

    # Reset counters
    with _CHUNKS_LOCK:
        _PROCESSED_CHUNKS = 0
        _EXPECTED_CHUNKS = (total_tokens + chunk_size - 1) // chunk_size

    # Clear flags
    _GENERATION_COMPLETE.clear()
    _CONSUMER_STARTED.clear()

    # Start our modified consumer thread
    if _consumer_t is None or not _consumer_t.is_alive():
        _consumer_t = threading.Thread(target=consumer_thread_with_collection)
        _consumer_t.start()
        # Wait for consumer to be ready
        _CONSUMER_STARTED.wait()

    carry = init_carry
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

    # Signal completion
    _GENERATION_COMPLETE.set()

    # Wait for all chunks to be processed
    print("Waiting for all chunks to be processed...")
    while True:
        with _CHUNKS_LOCK:
            if _PROCESSED_CHUNKS >= _EXPECTED_CHUNKS:
                break
        time.sleep(0.01)

    # Collect all tokens in order
    all_tokens = []
    with _TOKENS_LOCK:
        for i in range(chunk_count):
            if i in _COLLECTED_TOKENS:
                all_tokens.append(_COLLECTED_TOKENS[i])
    
    # Concatenate all chunks
    if all_tokens:
        # Shape: (total_generated, batch)
        concatenated = np.concatenate(all_tokens, axis=0)
        return carry, concatenated
    else:
        return carry, None

def extract_topic_keywords(response: str) -> List[str]:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'}
    words = response.lower().split()
    keywords = [w for w in words if len(w) > 4 and w not in stop_words]
    return keywords[:5]

def generate_user_turn(previous_response: str, turn_num: int, rng: np.random.RandomState) -> str:
    if turn_num == 0:
        return rng.choice(CONVERSATION_SEEDS)
    
    keywords = extract_topic_keywords(previous_response)
    if keywords:
        template = rng.choice(USER_TURN_TEMPLATES)
        keyword = rng.choice(keywords)
        return template.format(keyword)
    else:
        followups = [
            "Tell me more about that",
            "Can you elaborate?",
            "What else should I know?",
            "How does this work in practice?"
        ]
        return rng.choice(followups)

def create_conversation_batch(batch_size: int, turn_num: int, previous_responses: List[str], rng_seed: int) -> List[str]:
    rng = np.random.RandomState(rng_seed + turn_num)
    user_turns = []
    
    for i in range(batch_size):
        prev_response = previous_responses[i] if previous_responses else ""
        user_turn = generate_user_turn(prev_response, turn_num, rng)
        user_turns.append(user_turn)
    
    return user_turns

def format_conversation_context(conversation_history: List[Tuple[str, str]]) -> str:
    formatted = ""
    for i, (user, assistant) in enumerate(conversation_history):
        formatted += f"User: {user}\n"
        formatted += f"Assistant: {assistant}\n"
    return formatted

def run_multiturn_generation(
    model,
    tokenizer,
    config,
    state,
    cache,
    rope_cache,
    num_conversations: int = 8,
    max_turns: int = 4,
    max_tokens_per_turn: int = 256,
    temperature: float = 0.8,
    output_path: str = "conversations.jsonl"
):
    cache_length = config.cache_length
    chunk_length = config.chunk_length
    
    conversations = [[] for _ in range(num_conversations)]
    previous_responses = [""] * num_conversations
    
    for turn in range(max_turns):
        print(f"\nGenerating turn {turn + 1}/{max_turns}")
        
        user_turns = create_conversation_batch(
            num_conversations, 
            turn, 
            previous_responses,
            rng_seed=42
        )
        
        prompts = []
        for i, user_turn in enumerate(user_turns):
            if turn == 0:
                prompt = f"User: {user_turn}\nAssistant:"
            else:
                context = format_conversation_context(conversations[i])
                prompt = f"{context}User: {user_turn}\nAssistant:"
            prompts.append(prompt)
        
        input_ids = encode_text(prompts, tokenizer, add_special_tokens=True, return_tensors="np")
        
        padded_input_ids = np.pad(
            input_ids, 
            ((0, 0), (0, cache_length - input_ids.shape[1])), 
            constant_values=0
        )
        jax_input_ids = jnp.array(padded_input_ids, dtype=jnp.int32)
        
        t0 = time.time()
        prefill_cache = chunked_prefill(
            state=state,
            input_ids=jax_input_ids,
            model=model,
            cache=cache,
            rope_cache=rope_cache,
            config=config,
            chunk_length=chunk_length,
            cache_length=cache_length,
        )
        
        init_carry = setup_carry(
            state=state, 
            input_ids=input_ids, 
            prefill_cache=prefill_cache
        )
        
        # Use our modified function to collect tokens
        final_carry, generated_tokens = run_generation_and_collect_tokens(
            init_carry,
            model=model,
            rope_cache=rope_cache,
            config=config,
            total_tokens=max_tokens_per_turn,
            chunk_size=config.chunk_length
        )
        
        # Process generated tokens for each conversation
        responses = []
        if generated_tokens is not None:
            # generated_tokens shape: (total_generated, batch)
            for i in range(num_conversations):
                tokens = []
                for j in range(generated_tokens.shape[0]):
                    token = int(generated_tokens[j, i])
                    if token == EOS_ID or token == END_OF_TURN_ID:
                        break
                    tokens.append(token)
                
                # Decode tokens to text
                if tokens:
                    response = tokenizer.decode(tokens)
                else:
                    response = ""
                responses.append(response.strip())
        else:
            responses = [""] * num_conversations
        
        # Update conversation history
        for i in range(num_conversations):
            conversations[i].append((user_turns[i], responses[i]))
            previous_responses[i] = responses[i]
        
        print(f"Turn {turn + 1} completed in {time.time() - t0:.2f}s")
        
        # Print sample conversation
        if responses[0]:
            print(f"Sample response: {responses[0][:100]}...")
    
    # Save conversations with correct format
    with open(output_path, 'w') as f:
        for conv in conversations:
            conv_data = {
                "conversation": [],
                "num_turns": len(conv)
            }
            
            # Build conversation with alternating user/assistant messages
            for user_msg, assistant_msg in conv:
                conv_data["conversation"].append({
                    "role": "user", 
                    "content": user_msg
                })
                conv_data["conversation"].append({
                    "role": "assistant", 
                    "content": assistant_msg
                })
            
            f.write(json.dumps(conv_data) + '\n')
    
    print(f"\nSaved {num_conversations} conversations to {output_path}")
    return conversations

if __name__ == "__main__":
    conversations = run_multiturn_generation(
        model=model,
        tokenizer=tokenizer,
        config=config,
        state=state,
        cache=cache,
        rope_cache=rope_cache,
        num_conversations=2, # num_conversations must equal batch size
        max_turns=2,
        max_tokens_per_turn=64,
        temperature=0.8,
        output_path="synthetic_conversations.jsonl"
    )

    # Clean shutdown
    if _consumer_t and _consumer_t.is_alive():
        # Send poison pill to stop consumer thread
        _CHUNK_QUEUE.put(None)
        # Wait for thread to finish
        _consumer_t.join(timeout=1.0)
        if _consumer_t.is_alive():
            print("Warning: Consumer thread did not exit cleanly")

    print("Script completed successfully!")
    
    # Print first conversation as example
    if conversations and conversations[0]:
        print("\nExample conversation:")
        for turn_idx, (user, assistant) in enumerate(conversations[0]):
            print(f"\nTurn {turn_idx + 1}:")
            print(f"User: {user}")
            print(f"Assistant: {assistant}")

# %%
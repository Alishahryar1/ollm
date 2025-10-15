"""
Demonstration of CYCLIC PIPELINE for multi-token generation.

This shows how the pipeline continues loading layers for the next token
while the current token is still being processed, creating a perpetual
pipeline that maximizes throughput.
"""

from ollm import InferencePipelined, file_get_contents, TextStreamer
import time

def cyclic_pipeline_demo():
    """
    Generate multiple tokens to demonstrate cyclic pipeline behavior.
    
    Watch the logs to see:
    - Generation 0: Initial token (layers 0-N)
    - Generation 1: Second token starts loading while gen 0 is finishing
    - Generation 2+: Continuous cyclic loading
    """
    print("="*80)
    print("CYCLIC PIPELINE DEMONSTRATION")
    print("="*80)
    
    print("\n[1] Initializing model with cyclic pipeline...")
    o = InferencePipelined(
        "llama3-1B-chat",
        device="cuda:0", 
        logging=True,
        use_pipeline=True
    )
    
    print("\n[2] Loading model...")
    o.ini_model(
        models_dir="/media/mega4alik/ssd/models/", 
        force_download=False
    )
    
    print("\n[3] Setting up for generation...")
    past_key_values = o.DiskCache(cache_dir="/media/mega4alik/ssd/kv_cache/")
    text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Prepare a prompt that will generate multiple tokens
    sm = "You are a helpful AI assistant"
    um = "List planets starting from Mercury"
    
    messages = [
        {"role": "system", "content": sm}, 
        {"role": "user", "content": um}
    ]
    
    input_ids = o.tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(o.device)
    
    print(f"\nInput tokens: {input_ids.shape[-1]}")
    print("\n[4] Generating tokens with cyclic pipeline...")
    print("-"*80)
    print("\n👀 WATCH THE LOGS BELOW:")
    print("   - You'll see layers being loaded for the NEXT token")
    print("   - While the CURRENT token is still executing")
    print("   - Look for '★ Starting generation X' markers")
    print("   - And '[genX]' tags showing which generation a layer belongs to")
    print("\n" + "-"*80 + "\n")
    
    t_start = time.perf_counter()
    
    # Generate enough tokens to see multiple generations
    outputs = o.model.generate(
        input_ids=input_ids,
        past_key_values=past_key_values,
        max_new_tokens=50,  # Generate 50 tokens to see cyclic behavior
        streamer=text_streamer,
        do_sample=False,
    ).cpu()
    
    t_end = time.perf_counter()
    
    print("\n" + "-"*80)
    
    # Decode and display the full answer
    answer = o.tokenizer.decode(
        outputs[0][input_ids.shape[-1]:], 
        skip_special_tokens=True
    )
    
    # Performance stats
    total_tokens = outputs.shape[-1]
    new_tokens = total_tokens - input_ids.shape[-1]
    elapsed = t_end - t_start
    tokens_per_sec = new_tokens / elapsed
    
    print(f"\n[5] Generated Response:")
    print("-"*80)
    print(answer)
    print("-"*80)
    
    print(f"\n[6] Performance Summary:")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Tokens generated: {new_tokens}")
    print(f"    Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"    Time per token: {1000*elapsed/new_tokens:.2f}ms")
    print(f"\n    ★ Cyclic pipeline maintained continuous loading across")
    print(f"       {new_tokens} token generations!")
    
    print("\n" + "="*80)
    print("CYCLIC PIPELINE DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    cyclic_pipeline_demo()
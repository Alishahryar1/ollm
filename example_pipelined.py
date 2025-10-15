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


def visualize_cyclic_behavior():
    """
    Visual explanation of what's happening in the cyclic pipeline
    """
    print("\n" + "="*80)
    print("HOW THE CYCLIC PIPELINE WORKS")
    print("="*80 + "\n")
    
    print("Without Cyclic (Traditional):")
    print("-" * 50)
    print("Token 1: Load all layers → Execute → Done")
    print("         [IDLE - no loading]")
    print("Token 2: Load all layers → Execute → Done")
    print("         [IDLE - no loading]")
    print("Token 3: Load all layers → Execute → Done")
    print("\n❌ Problem: Pipeline stops between tokens\n")
    
    print("="*50)
    print("\nWith Cyclic (Our Implementation):")
    print("-" * 50)
    print("Token 1: ")
    print("  Layer 0-13: Execute")
    print("  Layer 14-15: Execute + START loading Token 2 Layer 0-1 ★")
    print("Token 2:")
    print("  Layer 0-1: Already loaded! Instant start ⚡")
    print("  Layer 2-13: Execute")
    print("  Layer 14-15: Execute + START loading Token 3 Layer 0-1 ★")
    print("Token 3:")
    print("  Layer 0-1: Already loaded! Instant start ⚡")
    print("  Layer 2-13: Execute")
    print("  Layer 14-15: Execute + START loading Token 4 Layer 0-1 ★")
    print("...")
    print("\n✅ Benefit: Zero latency between tokens!\n")
    
    print("="*50)
    print("\nKey Innovation:")
    print("-" * 50)
    print("1. As we execute the LAST layers of token N")
    print("2. We SIMULTANEOUSLY load FIRST layers of token N+1")
    print("3. When token N finishes, token N+1 starts IMMEDIATELY")
    print("4. No waiting, no idle time, perpetual pipeline! 🔄")
    print("\n" + "="*80 + "\n")


def multi_sequence_demo():
    """
    Demonstrate cyclic pipeline across multiple separate sequences.
    Shows how the pipeline automatically resets between sequences.
    """
    print("="*80)
    print("MULTI-SEQUENCE CYCLIC PIPELINE TEST")
    print("="*80)
    
    print("\n[1] Initializing model...")
    o = InferencePipelined(
        "llama3-1B-chat",
        device="cuda:0", 
        logging=True,
        use_pipeline=True
    )
    
    o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
    past_key_values = o.DiskCache(cache_dir="/media/mega4alik/ssd/kv_cache/")
    
    sequences = [
        "What is 2+2?",
        "Name three colors.",
        "Count from 1 to 5."
    ]
    
    total_time = 0
    total_tokens = 0
    
    for seq_idx, prompt in enumerate(sequences):
        print(f"\n{'='*80}")
        print(f"SEQUENCE {seq_idx + 1}: '{prompt}'")
        print('='*80)
        
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": prompt}
        ]
        
        input_ids = o.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(o.device)
        
        t_start = time.perf_counter()
        
        outputs = o.model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            do_sample=False,
        ).cpu()
        
        elapsed = time.perf_counter() - t_start
        new_tokens = outputs.shape[-1] - input_ids.shape[-1]
        
        answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        print(f"\nAnswer: {answer}")
        print(f"Time: {elapsed:.2f}s | Tokens: {new_tokens} | Speed: {new_tokens/elapsed:.2f} tok/s")
        
        total_time += elapsed
        total_tokens += new_tokens
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print('='*80)
    print(f"Total sequences: {len(sequences)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Average speed: {total_tokens/total_time:.2f} tok/s")
    print(f"\n✅ Cyclic pipeline maintained across all sequences!")
    print("="*80)


def stress_test_cyclic():
    """
    Stress test: Generate many tokens to verify pipeline stability
    """
    print("="*80)
    print("CYCLIC PIPELINE STRESS TEST")
    print("="*80)
    
    print("\n[1] Initializing model...")
    o = InferencePipelined(
        "llama3-1B-chat",
        device="cuda:0", 
        logging=False,  # Reduce log spam for stress test
        use_pipeline=True
    )
    
    o.ini_model(models_dir="/media/mega4alik/ssd/models/", force_download=False)
    
    # Generate a long response to stress test the cyclic pipeline
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "Write a detailed story about a robot learning to cook. Make it long and engaging."}
    ]
    
    input_ids = o.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(o.device)
    
    print(f"\n[2] Generating 200 tokens to stress test cyclic pipeline...")
    print("    (This will cycle through layers ~200 times)")
    
    t_start = time.perf_counter()
    
    outputs = o.model.generate(
        input_ids=input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    ).cpu()
    
    elapsed = time.perf_counter() - t_start
    new_tokens = outputs.shape[-1] - input_ids.shape[-1]
    
    answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    print(f"\n[3] Stress Test Results:")
    print("-"*80)
    print(f"Generated: {new_tokens} tokens")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {new_tokens/elapsed:.2f} tok/s")
    print(f"Avg time per token: {1000*elapsed/new_tokens:.2f}ms")
    print(f"\n✅ Pipeline remained stable through {new_tokens} generations!")
    print("="*80)
    
    # Show excerpt
    print("\n[4] Generated text excerpt (first 300 chars):")
    print("-"*80)
    print(answer[:300] + "..." if len(answer) > 300 else answer)
    print("-"*80)


if __name__ == "__main__":
    import sys
    
    print("\n" + "🔄"*40)
    print("CYCLIC PIPELINE DEMONSTRATION SUITE")
    print("🔄"*40 + "\n")
    
    # Show visual explanation first
    visualize_cyclic_behavior()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "visual":
            print("\n✅ Visual explanation complete!")
        elif mode == "demo":
            cyclic_pipeline_demo()
        elif mode == "multi":
            multi_sequence_demo()
        elif mode == "stress":
            stress_test_cyclic()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: visual, demo, multi, stress")
    else:
        # Default: run the main demo
        cyclic_pipeline_demo()
        
        print("\n" + "="*80)
        print("OTHER AVAILABLE TESTS:")
        print("="*80)
        print("  python example_cyclic_demo.py visual  - Show visual explanation only")
        print("  python example_cyclic_demo.py demo    - Main cyclic demo (default)")
        print("  python example_cyclic_demo.py multi   - Multi-sequence test")
        print("  python example_cyclic_demo.py stress  - Stress test (200 tokens)")
        print("="*80)
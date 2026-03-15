import argparse
import torch
import numpy as np
import os
import json
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not found. Falling back to Transformers generation (slow).")

# Reuse reasoning markers from main project
REASONING_MARKERS = {
    "en": [
        "let's think", "step by step", "reasoning", "analysis", 
        "chain-of-thought", "therefore", "first,", "second,", "in conclusion"
    ],
    "zh": [
        "让我们想", "一步一步", "步骤", "推理", "分析", 
        "首先", "其次", "因此", "综上", "结论"
    ]
}

def load_dataset(max_samples=None, seed=42):
    files = [
        "/root/workspace/train/data/medical/finetune/valid_en_1.json",
        "/root/workspace/train/data/medical/finetune/valid_zh_0.json"
    ]
    
    data = []
    for fpath in files:
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'prompt' not in item:
                            if 'instruction' in item and 'input' in item:
                                item['prompt'] = f"{item['instruction']}\n{item['input']}".strip()
                            elif 'instruction' in item:
                                item['prompt'] = item['instruction']
                            else:
                                continue
                        if 'id' not in item:
                            item['id'] = f"auto_{len(data)}"
                        data.append(item)
                    except:
                        continue
    
    # Simple shuffling
    import random
    random.seed(seed)
    random.shuffle(data)
    
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def check_leakage(text):
    text_lower = text.lower()
    
    # 1. Check for <think> tags
    if "<think>" in text_lower or "</think>" in text_lower:
        return True
        
    # 2. Check for reasoning markers
    for lang in REASONING_MARKERS:
        for marker in REASONING_MARKERS[lang]:
            if marker in text_lower:
                return True
                
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--template_name", required=True, choices=['plain', 'nothink_soft', 'nothink_strict'])
    parser.add_argument("--preset_name", required=True, choices=['PresetB'])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is required for this script but not found.")

    # Templates
    templates = {
        'plain': {
            'system': "You are a helpful assistant.",
            'user_suffix': ""
        },
        'nothink_soft': {
            'system': "You are a helpful assistant. Do not show your reasoning or internal thought process. Just provide the final answer.",
            'user_suffix': " (Do not show reasoning, only the answer)"
        },
        'nothink_strict': {
            'system': "You are a helpful assistant. PROHIBITION: You must NOT output any reasoning, thinking steps, analysis, or internal monologue. You must NOT use <think> tags. If you output any reasoning or analysis, it will be considered a failure. Output ONLY the final answer directly.",
            'user_suffix': " \n\n[CONSTRAINT: Output ONLY the answer. NO reasoning. NO <think> tags. NO analysis steps.]"
        }
    }
    
    presets = {
        'PresetB': {'temperature': 0.6, 'top_p': 0.8}
    }
    
    current_template = templates[args.template_name]
    current_preset = presets[args.preset_name]
    
    # ---------------------------------------------------------
    # Phase 1: Generation with vLLM
    # ---------------------------------------------------------
    print(f"\n=== Phase 1: vLLM Generation ({args.model_path}) ===")
    
    # 1. Prepare Data
    dataset = load_dataset(max_samples=args.max_samples, seed=args.seed)
    print(f"Loaded {len(dataset)} samples.")
    
    # 2. Format Prompts (Use basic formatting to avoid loading tokenizer if possible, 
    #    but better to use tokenizer to be safe with chat templates)
    #    We'll load tokenizer briefly to format, or just manual format. 
    #    To save memory, let's manual format as we did in original script fallback.
    
    prompts = []
    for item in dataset:
        prompt_text = f"{current_template['system']}\nUser: {item['prompt']}{current_template['user_suffix']}\nAssistant:"
        prompts.append(prompt_text)
        
    # 3. Initialize vLLM
    sampling_params = SamplingParams(
        temperature=current_preset['temperature'],
        top_p=current_preset['top_p'],
        max_tokens=args.max_new_tokens
    )
    
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(), # Use all visible GPUs
        gpu_memory_utilization=0.90,
        dtype="bfloat16" # Ensure BF16
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    # 4. Collect Results
    generated_texts = []
    for output in outputs:
        generated_texts.append(output.outputs[0].text)
        
    # 5. Cleanup vLLM (CRITICAL)
    print("Cleaning up vLLM...")
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("vLLM cleaned up.")

    # ---------------------------------------------------------
    # Phase 2: Activation Extraction with Transformers
    # ---------------------------------------------------------
    print(f"\n=== Phase 2: Transformers Extraction ({args.model_path}) ===")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # For batch generation-like alignment (though we just take last token)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    all_hidden_states = [] # List of [num_layers, hidden_dim]
    labels = []
    
    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    print(f"Processing {len(dataset)} samples in {num_batches} batches...")
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_gens = generated_texts[start_idx:end_idx]
        
        # Concat prompt + generation
        full_texts = [p + g for p, g in zip(batch_prompts, batch_gens)]
        
        # Tokenize
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
        # Extract last token hidden states
        # Since we use padding_side="left", the sequences are right-aligned.
        # The last token is simply at index -1 for all sequences in the batch.
        
        # outputs.hidden_states is a tuple of (layer_0, ..., layer_N)
        # Each layer tensor is [batch, seq_len, hidden]
        
        batch_layers_hs = [] # [batch, num_layers, hidden]
        
        num_layers = len(outputs.hidden_states)
        batch_size_curr = len(batch_prompts)
        
        # We want to pivot this to [batch, num_layers, hidden]
        # First, let's collect list of [batch, hidden] per layer
        
        for layer_idx, layer_tensor in enumerate(outputs.hidden_states):
            # layer_tensor: [batch, seq_len, hidden]
            # Take last token: [batch, -1, :] -> [batch, hidden]
            last_token_hs = layer_tensor[:, -1, :]
            
            # Convert to float32 to avoid BFloat16 error in numpy
            last_token_hs_f32 = last_token_hs.to(torch.float32).cpu()
            
            if layer_idx == 0:
                # Initialize batch container
                batch_layers_hs = [[] for _ in range(batch_size_curr)]
            
            for b in range(batch_size_curr):
                batch_layers_hs[b].append(last_token_hs_f32[b].numpy())
                
        # Add to global list
        for b in range(batch_size_curr):
            # stack layers -> [num_layers, hidden]
            all_hidden_states.append(np.stack(batch_layers_hs[b]))
            
            # Label
            is_leaky = check_leakage(batch_gens[b])
            labels.append(1 if is_leaky else 0)

    # Stack all samples
    final_hs = np.stack(all_hidden_states, axis=0) # [num_samples, num_layers, hidden_dim]
    final_labels = np.array(labels)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip('/'))
    save_name = f"model={model_name}__template={args.template_name}__preset={args.preset_name}"
    save_path = os.path.join(args.output_dir, f"{save_name}.npz")
    
    np.savez(save_path, hidden_states=final_hs, labels=final_labels)
    print(f"Saved activations to {save_path}")
    print(f"Shape: {final_hs.shape}, Labels mean: {final_labels.mean():.2f}")

if __name__ == "__main__":
    main()

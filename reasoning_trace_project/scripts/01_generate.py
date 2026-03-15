import argparse
import json
import os
import re
import time
import torch
import yaml
import random
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# --- Configuration & Constants ---

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

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_args():
    parser = argparse.ArgumentParser(description="Run reasoning trace reactivation experiments.")
    parser.add_argument("--models", nargs='+', required=True, help="List of model keys from models.yaml or 'all'")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--min_new_tokens", type=int, default=0, help="Min new tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model data type")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    return parser.parse_args()

def load_dataset(max_samples=None, shuffle=False, seed=42):
    # Hardcoded paths as per requirements
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
                        # Normalize prompt
                        if 'prompt' not in item:
                            if 'instruction' in item and 'input' in item:
                                item['prompt'] = f"{item['instruction']}\n{item['input']}".strip()
                            elif 'instruction' in item:
                                item['prompt'] = item['instruction']
                            else:
                                continue # Skip if no prompt can be formed
                        
                        # Ensure ID
                        if 'id' not in item:
                            item['id'] = f"auto_{len(data)}"
                            
                        data.append(item)
                    except:
                        continue
        else:
            print(f"Warning: File not found {fpath}")

    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        
    if max_samples is not None:
        data = data[:max_samples]
        
    return data

def check_think_tag(text):
    # Check for <think> ... </think> and variants
    # Case insensitive, optional spaces
    pattern = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if match:
        return True, match.group(0), match.start(), match.end()
    
    # Check for just <think> if closing tag is missing (open ended)
    pattern_open = re.compile(r"<\s*think\s*>", re.IGNORECASE)
    match_open = pattern_open.search(text)
    if match_open:
        return True, text[match_open.start():], match_open.start(), len(text)
        
    return False, "", -1, -1

def check_reasoning_markers(text):
    text_lower = text.lower()
    found = False
    first_idx = len(text)
    
    # English markers
    for m in REASONING_MARKERS['en']:
        idx = text_lower.find(m.lower())
        if idx != -1:
            found = True
            first_idx = min(first_idx, idx)
            
    # Chinese markers
    for m in REASONING_MARKERS['zh']:
        idx = text.find(m) # Case sensitive for Chinese usually fine, but text is mixed
        if idx != -1:
            found = True
            first_idx = min(first_idx, idx)
            
    if found:
        return True, first_idx
    return False, -1

def calculate_metrics(text, template_name):
    metrics = {
        "has_think_tag": False,
        "has_reasoning_markers": False,
        "leakage_chars": 0,
        "total_chars": len(text),
        "leakage_ratio": 0.0,
        "strict_fail": False,
        "answer_only_chars": 0
    }
    
    # 1. Check tags
    has_tag, tag_content, tag_start, tag_end = check_think_tag(text)
    metrics["has_think_tag"] = has_tag
    
    # 2. Check markers
    has_marker, marker_start = check_reasoning_markers(text)
    metrics["has_reasoning_markers"] = has_marker
    
    # 3. Leakage chars
    if has_tag:
        # Content inside tags (including tags themselves for character count usually? 
        # User definition: "取 think 段长度") -> Assume full tag block length
        metrics["leakage_chars"] = len(tag_content)
        
        # Calculate answer only chars (strip think block)
        answer_text = text[:tag_start] + text[tag_end:]
        metrics["answer_only_chars"] = len(answer_text)
        
    elif has_marker:
        # From first marker to end
        metrics["leakage_chars"] = len(text) - marker_start
        metrics["answer_only_chars"] = marker_start # Approximate
    else:
        metrics["leakage_chars"] = 0
        metrics["answer_only_chars"] = len(text)
        
    # 4. Ratio
    metrics["leakage_ratio"] = metrics["leakage_chars"] / max(metrics["total_chars"], 1)
    
    # 5. Strict fail
    if template_name == "nothink_strict":
        if metrics["has_think_tag"] or metrics["has_reasoning_markers"]:
            metrics["strict_fail"] = True
        # Check for structural patterns like "Step 1:" if not covered by markers
        # Adding a simple heuristic for "Step N:" or "1." at start of line
        if re.search(r"(?m)^(Step \d+|步骤 \d+|\d+\.)", text):
             metrics["strict_fail"] = True
             
    return metrics

def run_inference(args, models_config, presets_config, templates_config, dataset):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Filter models
    model_keys = list(models_config.keys()) if 'all' in args.models else args.models
    
    # Sharding logic: Select samples for this shard
    shard_indices = [i for i in range(len(dataset)) if i % args.num_shards == args.shard_id]
    shard_dataset = [dataset[i] for i in shard_indices]
    print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(shard_dataset)} samples.")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    output_file = os.path.join(args.output_dir, f"predictions_gpu{args.gpu_id}_shard{args.shard_id}.jsonl")
    meta_file = os.path.join(args.output_dir, f"run_meta_gpu{args.gpu_id}_shard{args.shard_id}.json")
    
    # Metadata
    run_meta = {
        "args": vars(args),
        "configs": {
            "models": models_config,
            "presets": presets_config,
            "templates": templates_config
        },
        "start_time": time.time(),
        "shard_samples": len(shard_dataset)
    }
    with open(meta_file, 'w') as f:
        json.dump(run_meta, f, indent=2)

    for model_key in model_keys:
        if model_key not in models_config:
            print(f"Model {model_key} not found in config. Skipping.")
            continue
            
        model_cfg = models_config[model_key]
        model_path = model_cfg['path']
        print(f"Loading model: {model_key} from {model_path}")
        
        torch_dtype = torch.bfloat16
        if args.dtype == "fp16":
            torch_dtype = torch.float16
        elif args.dtype == "fp32":
            torch_dtype = torch.float32

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # Check for pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": args.gpu_id}, # Force to specific GPU
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load model {model_key}: {e}")
            continue

        # Iterate templates
        for tmpl_name, tmpl_cfg in templates_config.items():
            # Iterate presets
            for preset_name, preset_cfg in presets_config.items():
                print(f"Running: {model_key} | {tmpl_name} | {preset_name}")
                
                for item in tqdm(shard_dataset, desc=f"{model_key}-{tmpl_name}-{preset_name}"):
                    # Construct Prompt
                    system_msg = tmpl_cfg.get('system', "")
                    user_msg = tmpl_cfg.get('user_prefix', "") + item['prompt'] + tmpl_cfg.get('user_suffix', "")
                    
                    messages = []
                    if system_msg:
                        messages.append({"role": "system", "content": system_msg})
                    messages.append({"role": "user", "content": user_msg})
                    
                    # Apply template
                    try:
                        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    except Exception as e:
                        # Fallback if template fails or not supported
                        prompt_text = ""
                        for m in messages:
                            prompt_text += f"<|{m['role']}|>\n{m['content']}\n"
                        prompt_text += "<|assistant|>\n"
                    
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                    
                    # Generate
                    gen_kwargs = {
                        "max_new_tokens": args.max_new_tokens,
                        "min_new_tokens": args.min_new_tokens,
                        "repetition_penalty": args.repetition_penalty,
                        "do_sample": preset_cfg['do_sample'],
                        "temperature": preset_cfg['temperature'],
                        "top_p": preset_cfg['top_p'],
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    }
                    
                    # Set seed
                    set_seed(preset_cfg['seed']) # Transformers set_seed handles torch, cuda, numpy, random
                        
                    try:
                        with torch.no_grad():
                            outputs = model.generate(**inputs, **gen_kwargs)
                            
                        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                        # CRITICAL: Do NOT skip special tokens to catch <think> if it's special, and no post-processing
                        output_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM Error for item {item['id']}. Skipping and clearing cache.")
                        torch.cuda.empty_cache()
                        output_text = "ERROR_OOM"
                    except Exception as e:
                        print(f"Error generating for item {item['id']}: {e}")
                        output_text = f"ERROR_GEN: {str(e)}"
                    
                    # Metrics
                    metrics = calculate_metrics(output_text, tmpl_name)
                    
                    # Result Object
                    result = {
                        "id": item['id'],
                        "prompt": item['prompt'], # Original prompt
                        "output": output_text,    # User requested key "output"
                        "output_text": output_text, # Keep old key for compatibility just in case
                        "model": model_key,       # User requested key "model"
                        "model_id": model_key,    # Keep old key
                        "template": tmpl_name,    # User requested key "template"
                        "template_name": tmpl_name, # Keep old key
                        "preset": preset_name,
                        "preset_name": preset_name,
                        "thinking": metrics["has_think_tag"], # User requested key "thinking"
                        "seed": preset_cfg['seed'],
                        "temperature": preset_cfg['temperature'],
                        "top_p": preset_cfg['top_p'],
                        "do_sample": preset_cfg['do_sample'],
                        "max_new_tokens": args.max_new_tokens,
                        **metrics
                    }
                    
                    # Append to file immediately
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        
    # Update meta with end time
    with open(meta_file, 'r') as f:
        run_meta = json.load(f)
    run_meta['end_time'] = time.time()
    with open(meta_file, 'w') as f:
        json.dump(run_meta, f, indent=2)

if __name__ == "__main__":
    args = setup_args()
    
    models = load_yaml(os.path.join(args.config_dir, "models.yaml")).get('models', {})
    presets = load_yaml(os.path.join(args.config_dir, "presets.yaml")).get('presets', {})
    templates = load_yaml(os.path.join(args.config_dir, "templates.yaml")).get('templates', {})
    
    if not models:
        print("Error: No models found in models.yaml (check indentation or root key 'models')")
        exit(1)
    
    data = load_dataset(args.max_samples, args.shuffle, args.seed)
    
    run_inference(args, models, presets, templates, data)

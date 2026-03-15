import argparse
import os
import csv
import json
import torch
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vllm import LLM, SamplingParams
from utils import load_config, get_logit_bias_for_markers

def load_mcq_data(path, dataset_type, N, seed=42):
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    random.seed(seed)
    random.shuffle(data)
    data = data[:N]
    
    formatted = []
    for item in data:
        # MMLU/MedQA format: question, choices (list or dict), answer (A/B/C/D)
        q = item.get('question', '')
        choices = item.get('choices', [])
        ans = item.get('answer', '')
        
        # Format choices
        options_text = ""
        labels = ['A', 'B', 'C', 'D', 'E']
        
        if isinstance(choices, list):
            for i, c in enumerate(choices):
                if i < len(labels):
                    options_text += f"{labels[i]}. {c}\n"
        elif isinstance(choices, dict):
            # Sometimes choices are dicts
            for k, v in choices.items():
                options_text += f"{k}. {v}\n"
        
        prompt = f"Question: {q}\n{options_text}Answer:"
        formatted.append({
            'prompt': prompt,
            'gold': ans,
            'type': dataset_type,
            'id': item.get('id', '')
        })
    return formatted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_alias", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    model_path = config['models'][args.model_alias]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset paths exist
    mmlu_path = config['dataset'].get('mmlu_path')
    medqa_path = config['dataset'].get('medqa_path')
    
    if not mmlu_path or not os.path.exists(mmlu_path):
        print(f"MMLU path not found: {mmlu_path}")
        return
    if not medqa_path or not os.path.exists(medqa_path):
        print(f"MedQA path not found: {medqa_path}")
        return

    print(f"[{args.model_alias}] Starting Marker-Irrelevant Supervision Experiment...")

    # Load Data
    N = config['dataset']['N']
    seed = config['seed']
    mmlu_data = load_mcq_data(mmlu_path, "MMLU", N, seed)
    medqa_data = load_mcq_data(medqa_path, "MedQA", N, seed)
    
    all_data = mmlu_data + medqa_data
    
    # Init LLM
    try:
        llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True,
                  gpu_memory_utilization=0.9, max_model_len=4096)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    markers = config['markers']
    gammas = [0, 1] # Baseline vs Intervention
    
    results_detail = []
    results_summary = []
    
    for gamma in gammas:
        print(f"Running Gamma={gamma}...")
        logit_bias = get_logit_bias_for_markers(model_path, markers, gamma, scale=5.0)
        
        # Sampling for MCQ: low temp, return logprobs
        sp = SamplingParams(
            temperature=0.0, # Greedy for MCQ
            max_tokens=5,    # Short answer
            logit_bias=logit_bias,
            logprobs=5,      # Top 5 logprobs
            detokenize=True
        )
        
        prompts = [d['prompt'] for d in all_data]
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        # Analyze
        current_preds = []
        current_confs = []
        current_correct = []
        current_types = []
        
        for i, o in enumerate(outputs):
            gold = all_data[i]['gold']
            dtype = all_data[i]['type']
            
            # Get logprobs of the first token
            # We look for A, B, C, D in the top-k logprobs
            first_token_logprobs = o.outputs[0].logprobs[0] # Dict {id: logprob}
            
            # Map token text to A/B/C/D
            # Note: Tokenizer might produce " A" or "A" or " A" (sentencepiece)
            # We scan the top logprobs for the option letters
            option_probs = {'A': -999, 'B': -999, 'C': -999, 'D': -999}
            
            for tid, lp in first_token_logprobs.items():
                token_text = o.outputs[0].prompt_token_ids # Wait, we need detokenized text of this token
                # vLLM logprobs keys are token IDs. We need to decode them or trust the text if available?
                # Actually o.outputs[0].logprobs is {token_id: Logprob}
                # We can't easily get text for just this ID without tokenizer.
                # BUT, vLLM 0.15 might return SampleLogprob object which has `decoded_token`?
                # Let's assume standard dict {int: float}.
                # We should have passed the tokenizer or used the output text.
                pass
            
            # Fallback: Just look at generated text
            gen_text = o.outputs[0].text.strip().upper()
            # Simple parsing: take first letter
            pred = gen_text[0] if gen_text else "X"
            
            # Check correctness
            is_correct = 1 if pred.startswith(gold) else 0
            
            # Compute Confidence: Exp(logprob) of the generated token
            # This is a rough proxy.
            try:
                # Get logprob of the chosen token (first one)
                chosen_id = o.outputs[0].token_ids[0]
                conf = o.outputs[0].logprobs[0][chosen_id]
            except:
                conf = -10.0
                
            current_preds.append(pred)
            current_confs.append(conf)
            current_correct.append(is_correct)
            current_types.append(dtype)
            
            results_detail.append({
                "model": args.model_alias,
                "gamma": gamma,
                "type": dtype,
                "id": all_data[i]['id'],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "confidence": conf
            })
            
        # Per-dataset metrics
        for dtype in ["MMLU", "MedQA"]:
            indices = [k for k, t in enumerate(current_types) if t == dtype]
            if not indices:
                continue
                
            subset_correct = [current_correct[k] for k in indices]
            subset_conf = [current_confs[k] for k in indices]
            
            acc = sum(subset_correct) / len(subset_correct)
            
            # AUC (Confidence -> Correctness)
            if len(set(subset_correct)) > 1:
                auc = roc_auc_score(subset_correct, subset_conf)
            else:
                auc = 0.5
                
            results_summary.append({
                "model": args.model_alias,
                "gamma": gamma,
                "dataset": dtype,
                "accuracy": acc,
                "auc_conf_correct": auc
            })
            print(f"  {dtype} Gamma={gamma}: Acc={acc:.2f}, AUC={auc:.2f}")

    # Cleanup
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Save
    csv_path = os.path.join(output_dir, f"marker_irrelevant_{args.model_alias}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
        writer.writeheader()
        writer.writerows(results_summary)
    print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()

import os
import json
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
import yaml

# Add current dir to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import build_marker_token_ids

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_mcq_data(path, dataset_type, N, seed=42):
    """
    Load MCQ data from MMLU or MedQA jsonl files.
    Returns list of dicts with 'prompt', 'gold', 'type', 'id'.
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    # Shuffle and select N
    import random
    random.seed(seed)
    random.shuffle(data)
    data = data[:N]
    
    formatted = []
    for item in data:
        q = item.get('question', '')
        choices = item.get('choices', [])
        ans = item.get('answer', '')
        
        # Heuristic for Reasoning Demand (Exp 5B)
        is_high_demand = 0
        if len(q) > 200 or any(char.isdigit() for char in q) or "calculate" in q.lower():
            is_high_demand = 1
            
        options_text = ""
        labels = ['A', 'B', 'C', 'D', 'E']
        if isinstance(choices, list):
            for i, c in enumerate(choices):
                if i < len(labels):
                    options_text += f"{labels[i]}. {c}\n"
        
        # Answer-only prompt template (Marker-Forbidden)
        prompt = f"Question: {q}\n{options_text}Answer ONLY with the correct option letter (A, B, C, D). Do not explain.\nAnswer:"
        
        formatted.append({
            'prompt': prompt,
            'gold': ans,
            'type': dataset_type,
            'id': item.get('id', ''),
            'y_demand': is_high_demand
        })
    return formatted

def build_u_corr_and_rand(args):
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    
    model_path = args.model_path
    dataset_N = config['dataset']['N']
    mmlu_path = config['dataset']['mmlu_path']
    medqa_path = config['dataset']['medqa_path']
    
    print(f"Loading model: {model_path}")
    # Initialize LLM with raw_logits mode as in Exp5
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
        max_logprobs=200000, # Increased to allow full vocab request
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load Datasets
    datasets = {
        'mmlu': load_mcq_data(mmlu_path, 'mmlu', dataset_N),
        'medqa': load_mcq_data(medqa_path, 'medqa', dataset_N)
    }
    
    # Collect vectors
    vectors = []
    labels = []
    
    # Check for markers (for reporting only)
    markers = config.get('markers', [])
    marker_ids = build_marker_token_ids(model_path, markers)
    marker_hit_count = 0
    total_samples = 0
    
    # K steps for averaging
    K = 8
    
    print("Collecting vectors for u_corr construction...")
    
    for d_name, d_data in datasets.items():
        for item in tqdm(d_data):
            prompt = item['prompt']
            gold = item['gold']
            
            # Sampling params
            # We need full logprobs (vocab_size) to build dense u
            # We generate short answer to check correctness
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
                logprobs=vocab_size, # Request full logits/logprobs
                stop=["\n", "Question:", "User:"]
            )
            
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text.strip()
            
            # Check correctness
            pred_correct = 0
            if gold.lower() in output_text.lower():
                pred_correct = 1
            
            # Check marker hit (just for reporting)
            for m in markers:
                if m in output_text:
                    marker_hit_count += 1
                    break
            
            # Extract z_t at t=1..K
            # logprobs is List[Dict[int, float]]
            all_step_logprobs = outputs[0].outputs[0].logprobs
            
            # Aggregate first K steps
            steps_to_use = all_step_logprobs[:K]
            if not steps_to_use:
                continue

            # We sum vectors then divide by count
            avg_vec = np.zeros(vocab_size, dtype=np.float32)
            count = 0
            
            for step_lp in steps_to_use:
                if not step_lp: continue
                # step_lp is dict token_id -> logprob
                # Convert to dense
                step_vec = np.zeros(vocab_size, dtype=np.float32)
                for tid, val in step_lp.items():
                    if tid < vocab_size:
                        step_vec[tid] = val.logprob
                avg_vec += step_vec
                count += 1
            
            if count > 0:
                avg_vec /= count
                vectors.append(avg_vec)
                labels.append(pred_correct)
                total_samples += 1
            
    # Compute u_corr
    vectors = np.array(vectors) # (Total_N, Vocab)
    labels = np.array(labels)
    
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    print(f"Total samples: {total_samples}")
    print(f"Correct: {len(pos_indices)}")
    print(f"Incorrect: {len(neg_indices)}")
    print(f"Marker Hit Rate: {marker_hit_count/total_samples if total_samples else 0:.4f}")
    
    diagnostics = {
        "n_pos": int(len(pos_indices)),
        "n_neg": int(len(neg_indices)),
        "marker_hit_rate": float(marker_hit_count/total_samples) if total_samples else 0.0
    }

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print("Warning: One class is empty. u_corr will be zeros.")
        u_corr = np.zeros(vocab_size, dtype=np.float32)
        diagnostics["norm_diff"] = 0.0
        diagnostics["separation"] = 0.0
    else:
        mu_pos = np.mean(vectors[pos_indices], axis=0)
        mu_neg = np.mean(vectors[neg_indices], axis=0)
        diff = mu_pos - mu_neg
        norm = np.linalg.norm(diff)
        print(f"Norm of (mu_pos - mu_neg): {norm}")
        
        diagnostics["norm_diff"] = float(norm)
        
        if norm > 0:
            u_corr = diff / norm
        else:
            u_corr = diff
            
        # Separation on training set
        # score_i = mean_{t<=K}(u_corr^T z_t) = u_corr^T v_i
        scores = np.dot(vectors, u_corr)
        mean_score_pos = np.mean(scores[pos_indices])
        mean_score_neg = np.mean(scores[neg_indices])
        separation = mean_score_pos - mean_score_neg
        print(f"Separation (pos - neg): {separation}")
        diagnostics["mean_pos_score"] = float(mean_score_pos)
        diagnostics["mean_neg_score"] = float(mean_score_neg)
        diagnostics["train_sep"] = float(separation) # True training separation
        diagnostics["delta_norm"] = float(norm) # ||mu_pos - mu_neg||


    # Save u_corr
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(__file__), "..", "outputs", "exp6")
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_safe = args.model_name.replace("/", "_")
    u_corr_path = os.path.join(output_dir, f"u_corr_{model_name_safe}.npy")
    np.save(u_corr_path, u_corr)
    print(f"Saved u_corr to {u_corr_path}")
    
    # Generate and save u_rand
    rng = np.random.default_rng(42)
    u_rand = rng.standard_normal(vocab_size).astype(np.float32)
    u_rand = u_rand / np.linalg.norm(u_rand)
    
    u_rand_path = os.path.join(output_dir, f"u_rand_{model_name_safe}.npy")
    np.save(u_rand_path, u_rand)
    print(f"Saved u_rand to {u_rand_path}")
    
    # Cosine similarity
    cos_sim = float(np.dot(u_corr, u_rand))
    print(f"Cos sim (u_corr, u_rand): {cos_sim}")
    diagnostics["cos_sim_rand"] = cos_sim
    
    # Save diagnostics
    diag_path = os.path.join(output_dir, f"diagnostics_u_corr_{model_name_safe}.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {diag_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    build_u_corr_and_rand(args)


import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
import yaml
from sklearn.decomposition import PCA

# Add Exp6 dir to path for utils
EXP6_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments_markerfree", "exp6_correctness_defined_u")
sys.path.append(EXP6_DIR)
from utils import build_marker_token_ids

def load_config(config_path):
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

import gc
import torch

def build_Uk(args):
    # Load config from Exp6
    config_path = os.path.join(EXP6_DIR, "config.yaml")
    config = load_config(config_path)
    
    model_path = args.model_path
    dataset_N = args.N_dir if args.N_dir else config['dataset']['N']
    mmlu_path = config['dataset']['mmlu_path']
    medqa_path = config['dataset']['medqa_path']
    
    print(f"Loading model: {model_path}")
    # Initialize LLM with raw_logits mode
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
        max_logprobs=200000, # Full vocab
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load Datasets
    datasets = {
        'mmlu': load_mcq_data(mmlu_path, 'mmlu', dataset_N, seed=args.seed),
        'medqa': load_mcq_data(medqa_path, 'medqa', dataset_N, seed=args.seed)
    }
    
    # Collect vectors
    vectors = []
    labels = []
    
    # K steps for averaging (Same as Exp6)
    K = 8
    
    print("Collecting vectors...")
    
    for d_name, d_data in datasets.items():
        for item in tqdm(d_data):
            prompt = item['prompt']
            gold = item['gold']
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
                logprobs=vocab_size,
                stop=["\n", "Question:", "User:"]
            )
            
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text.strip()
            
            # Check correctness
            pred_correct = 0
            if gold.lower() in output_text.lower():
                pred_correct = 1
            
            # Extract z_t at t=1..K
            all_step_logprobs = outputs[0].outputs[0].logprobs
            steps_to_use = all_step_logprobs[:K]
            if not steps_to_use:
                continue

            avg_vec = np.zeros(vocab_size, dtype=np.float32)
            count = 0
            
            for step_lp in steps_to_use:
                if not step_lp: continue
                # step_lp is dict token_id -> logprob
                
                # Optimized extraction
                # Extract keys and values directly
                step_ids = list(step_lp.keys())
                step_vals = [x.logprob for x in step_lp.values()]
                
                # Create sparse update (much faster than iterating)
                # But since we use dense avg_vec, we can just index
                step_vec = np.zeros(vocab_size, dtype=np.float32)
                
                # Handle potential out of bound indices if vocab_size mismatch
                valid_indices = [i for i, idx in enumerate(step_ids) if idx < vocab_size]
                if valid_indices:
                    valid_ids = [step_ids[i] for i in valid_indices]
                    valid_vals = [step_vals[i] for i in valid_indices]
                    step_vec[valid_ids] = valid_vals
                
                avg_vec += step_vec
                count += 1
            
            if count > 0:
                avg_vec /= count
                vectors.append(avg_vec)
                labels.append(pred_correct)
            
    vectors = np.array(vectors) # (Total_N, Vocab)
    labels = np.array(labels)
    
    # Free VLLM resources
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("VLLM resources released.")

    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    print(f"Total samples: {len(vectors)}")
    print(f"Correct: {len(pos_indices)}")
    print(f"Incorrect: {len(neg_indices)}")
    
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        print("Error: One class is empty. Cannot compute u_corr.")
        return

    # 1. Compute u_corr (Rank 1 direction)
    mu_pos = np.mean(vectors[pos_indices], axis=0)
    mu_neg = np.mean(vectors[neg_indices], axis=0)
    diff = mu_pos - mu_neg
    norm = np.linalg.norm(diff)
    if norm > 0:
        u_corr = diff / norm
    else:
        u_corr = diff
        
    print(f"Computed u_corr. Norm: {norm}")
    
    k = args.k
    
    if k == 1:
        Uk = u_corr.reshape(-1, 1)
        explained_variance = [1.0] # Dummy
    else:
        # Residual PCA
        print(f"Computing Residual PCA for k={k}...")
        # Project data onto orthogonal complement of u_corr
        # V_perp = V - (V . u) * u^T
        projections = np.dot(vectors, u_corr) # (N,)
        V_perp = vectors - np.outer(projections, u_corr)
        
        # Check for NaNs
        if np.isnan(V_perp).any():
            print("Warning: V_perp contains NaNs. Replacing with 0.")
            V_perp = np.nan_to_num(V_perp)
            
        # PCA on V_perp (Using PyTorch SVD to avoid sklearn/vllm contention)
        print(f"Running PCA (SVD) on shape {V_perp.shape} using PyTorch...")
        try:
            # Convert to torch
            V_perp_torch = torch.from_numpy(V_perp).float()
            
            # SVD
            # V_perp = U S V^T
            # PCs are rows of V^T, i.e., columns of V
            # torch.linalg.svd returns U, S, Vh
            # Vh is V^T, so its rows are the principal components
            U_svd, S_svd, Vh_svd = torch.linalg.svd(V_perp_torch, full_matrices=False)
            
            # Top k-1 components
            pcs = Vh_svd[:k-1, :].T.numpy() # (D, k-1)
            
            # Explained variance (S^2 / (n-1))
            explained_variance_ = (S_svd[:k-1] ** 2 / (V_perp.shape[0] - 1)).numpy()
            total_var = torch.sum(S_svd ** 2 / (V_perp.shape[0] - 1)).item()
            explained_variance_ratio_ = explained_variance_ / total_var
            
            print("PCA fit done.")
        except Exception as e:
            print(f"PCA failed: {e}")
            # Fallback to random or zero if PCA fails?
            print("Falling back to random directions for remaining components.")
            pcs = np.random.randn(vectors.shape[1], k-1)
            explained_variance_ratio_ = [0.0] * (k-1)
        
        # Stack: [u_corr, pcs]
        Uk_raw = np.column_stack([u_corr, pcs]) # (D, k)
        
        # Orthonormalize via QR
        # Uk_raw = Q * R
        Q, R = np.linalg.qr(Uk_raw)
        
        # Fix signs to match Uk_raw directions
        # dot(Q[:, i], Uk_raw[:, i]) should be > 0
        for i in range(k):
            if np.dot(Q[:, i], Uk_raw[:, i]) < 0:
                Q[:, i] = -Q[:, i]
                
        Uk = Q
        explained_variance = [1.0] + list(explained_variance_ratio_)
        
    # Check orthogonality
    gram = np.dot(Uk.T, Uk)
    print("Gram matrix (should be Identity):")
    print(np.round(gram, 2))
    
    # Save
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    # Ensure model name doesn't contain path separators
    model_name_safe = args.model_name.replace("/", "_")
    output_path = os.path.join(output_dir, f"Uk_{model_name_safe}_k{k}.npz")
    
    np.savez(output_path, Uk=Uk, explained_variance=explained_variance, 
             model_name=args.model_name, k=k, seed=args.seed, N_dir=dataset_N)
    print(f"Saved Uk to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="DeepSeek")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--N_dir", type=int, default=500)
    args = parser.parse_args()
    
    build_Uk(args)

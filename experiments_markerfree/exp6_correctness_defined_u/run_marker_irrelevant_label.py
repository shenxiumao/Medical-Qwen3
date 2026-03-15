import os
import json
import glob

os.environ["VLLM_USE_V1"] = "0"

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from vllm import LLM, SamplingParams

# Import utils from the same folder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import build_marker_token_ids

def vector_to_logit_bias(u_vector, gamma, scale=5.0):
    if gamma == 0:
        return {}
    bias = -gamma * scale * u_vector
    # Filter small values to avoid huge dict overhead if possible, 
    # but u_corr is dense. 1e-6 threshold.
    indices = np.where(np.abs(bias) > 1e-6)[0]
    d = {}
    for idx in indices:
        d[int(idx)] = float(bias[idx])
    return d

def load_u_vectors(model_name, output_dir):
    model_name_safe = model_name.replace("/", "_")
    u_corr_path = os.path.join(output_dir, f"u_corr_{model_name_safe}.npy")
    u_rand_path = os.path.join(output_dir, f"u_rand_{model_name_safe}.npy")
    
    if not os.path.exists(u_corr_path):
        raise FileNotFoundError(f"u_corr not found at {u_corr_path}. Run build_u_corr.py first.")
        
    u_corr = np.load(u_corr_path)
    u_rand = np.load(u_rand_path)
    return u_corr, u_rand


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
    for idx, item in enumerate(data):
        q = item.get('question', '')
        choices = item.get('choices', [])
        ans = item.get('answer', '')
        
        # Heuristic for Reasoning Demand (Exp 5B)
        # High demand: Long question (>200 chars) OR contains numbers OR has "calculate"
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
        
        # Ensure id is unique and non-null (use idx if needed)
        sample_id = item.get('id')
        if sample_id is None or sample_id == "":
            sample_id = idx
        
        formatted.append({
            'prompt': prompt,
            'gold': ans,
            'type': dataset_type,
            'id': sample_id,
            'y_demand': is_high_demand
        })
    return formatted

def build_u_vectors(marker_ids, vocab_size, seed=42):
    u_real = np.zeros(vocab_size, dtype=np.float32)
    if marker_ids:
        valid_ids = [mid for mid in marker_ids if 0 <= int(mid) < vocab_size]
        if valid_ids:
            u_real[valid_ids] = 1.0
    norm = np.linalg.norm(u_real)
    if norm > 0:
        u_real = u_real / norm

    rng = np.random.default_rng(seed)
    u_rand = rng.standard_normal(vocab_size).astype(np.float32)
    u_rand = u_rand / np.linalg.norm(u_rand)
    return u_real, u_rand

def extract_features_from_logprobs(sample_logprobs, u_real, u_rand):
    if not sample_logprobs:
        return [0.0] * 6

    step_real = []
    step_rand = []
    step_entropy = []
    step_margin = []

    for pos in sample_logprobs:
        if pos is None:
            continue
        if isinstance(pos, dict):
            token_ids = np.array(list(pos.keys()), dtype=np.int32)
            logits = np.array([pos[t].logprob for t in token_ids], dtype=np.float32)
        else:
            token_ids = np.array(list(pos.keys()), dtype=np.int32)
            logits = np.array([pos[t].logprob for t in token_ids], dtype=np.float32)

        if token_ids.size == 0:
            continue
        max_idx = min(u_real.shape[0], u_rand.shape[0])
        valid_mask = (token_ids >= 0) & (token_ids < max_idx)
        if not np.any(valid_mask):
            continue
        token_ids = token_ids[valid_mask]
        logits = logits[valid_mask]

        real_dot = float(np.dot(u_real[token_ids], logits))
        rand_dot = float(np.dot(u_rand[token_ids], logits))
        step_real.append(real_dot)
        step_rand.append(rand_dot)

        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        step_entropy.append(float(entropy))

        if logits.size >= 2:
            top2 = np.partition(logits, -2)[-2:]
            margin = float(np.max(top2) - np.min(top2))
        else:
            margin = 0.0
        step_margin.append(margin)

    if not step_real:
        return [0.0] * 6

    f1 = float(np.mean(step_real))
    f2 = float(np.max(step_real))
    f1_rand = float(np.mean(step_rand))
    f2_rand = float(np.max(step_rand))
    f3 = float(np.mean(step_entropy)) if step_entropy else 0.0
    f4 = float(step_margin[0]) if step_margin else 0.0
    return [f1, f2, f3, f4, f1_rand, f2_rand]

def run_experiment(args):
    # Load Config
    config_path = args.config_path if args.config_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)
    
    model_path = args.model_path
    dataset_N = config['dataset']['N']
    mmlu_path = config['dataset']['mmlu_path']
    medqa_path = config['dataset']['medqa_path']
    markers = config.get('markers', config.get('dataset', {}).get('markers', []))
    gamma_list = config['gamma_grid']
    # Increase logprobs to ensure we catch option tokens
    logprobs_k = 500  # Increased from min(..., 20) to capture option logits
    
    # Initialize LLM
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
        max_logprobs=500, # Increased to allow capturing option logits
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Load U vectors (Exp6 change)
    output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(__file__), "..", "outputs", "exp6")
    u_real, u_rand = load_u_vectors(args.model_name, output_dir)
    
    # Select intervention vector
    if args.direction_type == 'real':
        u_intervention = u_real
    elif args.direction_type == 'rand':
        u_intervention = u_rand
    else:
        raise ValueError(f"Unknown direction_type: {args.direction_type}")
    
    # Load Datasets
    datasets = {
        'mmlu': load_mcq_data(mmlu_path, 'mmlu', dataset_N),
        'medqa': load_mcq_data(medqa_path, 'medqa', dataset_N)
    }
    
    # Pre-compute option IDs
    option_chars = ["A", "B", "C", "D"]
    # Be careful with tokenization: some models token "A" is different from " A"
    # We will look for both
    option_ids = {}
    option_ids_sp = {} # For space-prefixed tokens
    
    for char in option_chars:
        # 1. Clean "A"
        t1 = tokenizer.encode(char, add_special_tokens=False)
        if t1: 
            option_ids[char] = t1[-1] 
        else:
            option_ids[char] = None
            
        # 2. Space " A"
        t2 = tokenizer.encode(" " + char, add_special_tokens=False)
        if t2: 
            option_ids_sp[char] = t2[-1]
        else:
            option_ids_sp[char] = None
    
    results = []
    
    for gamma in gamma_list:
        print(f"Running gamma={gamma} with direction={args.direction_type}")

        
        # Exp6: Use vector_to_logit_bias
        logit_bias = vector_to_logit_bias(u_intervention, gamma, scale=5.0)
        
        for d_name, d_data in datasets.items():
            print(f"Processing {d_name}...")
            
            for item in tqdm(d_data):
                prompt = item['prompt']
                gold = item['gold']
                y_demand = item['y_demand']
                
                # Setup Sampling Params
                sampling_params = SamplingParams(
                    temperature=0.0, # Deterministic for answer-only
                    max_tokens=10,   # Short answer
                    logit_bias=logit_bias,
                    logprobs=logprobs_k,
                    flat_logprobs=False,
                    stop=["\n", "Question:", "User:"] # Stop early
                )
                
                # Generate
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                output_text = outputs[0].outputs[0].text.strip()
                
                # Determine Correctness
                pred_correct = 0
                if gold.lower() in output_text.lower():
                     pred_correct = 1
                
                # Check Marker Hit (Exp6)
                marker_hit = 0
                for m in markers:
                    if m in output_text:
                        marker_hit = 1
                        break

                # New Metrics Calculation (Exp6+)
                # 1. Refusal Rate
                refusal_patterns = ["I can't", "I cannot", "I'm sorry", "I am sorry", "As an AI"]
                refusal_detected = 0
                for p in refusal_patterns:
                    if p.lower() in output_text.lower():
                        refusal_detected = 1
                        break
                
                # 2. Format Violation
                # Expected: Single letter A/B/C/D or very short phrase starting with it.
                # Loose check: if no A/B/C/D found, or text is very long (>50 chars) and not a clear answer.
                format_violation = 0
                ans_chars = ['A', 'B', 'C', 'D']
                found_ans = False
                for char in ans_chars:
                    if char in output_text.upper(): # loose check
                        found_ans = True
                        break
                if not found_ans or len(output_text) > 100:
                    format_violation = 1

                # Extract Features & Logic Metrics
                sample_logprobs = outputs[0].outputs[0].logprobs
                feats = extract_features_from_logprobs(sample_logprobs, u_real, u_rand)
                
                # 3. Choice Margin & Entropy (from ACTUAL DECISION STEP)
                choice_margin = 0.0
                answer_entropy = 0.0
                
                # Variables for option-specific logits
                logit_dec_map = {'A': -np.inf, 'B': -np.inf, 'C': -np.inf, 'D': -np.inf}
                decision_step = -1
                option_logits_logged = False
                
                # Identify decision step t*
                token_ids = outputs[0].outputs[0].token_ids
                
                for t_idx, tid in enumerate(token_ids):
                    # Decode single token
                    token_text = tokenizer.decode([tid], skip_special_tokens=True)
                    # Check if token matches answer pattern: "A", " A", "A.", "A)", "(A)"
                    # Regex: optional space, optional paren, [ABCD], optional paren/dot
                    # Actually user said: Exact single token "A"/"B"/"C"/"D" Or prefixes like "A.", "A)", "(A)"
                    # We check stripped version or use regex
                    import re
                    # Pattern: starts with optional space, optional open paren, capture [ABCD], optional close paren/dot, end of string (or just looks like answer)
                    # We need to be careful not to match "And" as "A"
                    # But token_text is just one token.
                    # If token is " And", it doesn't match.
                    # If token is " A", it matches.
                    # If token is "A.", it matches.
                    
                    match = re.match(r"^\s*\(?([ABCD])\)?\.?$", token_text)
                    if match:
                        decision_step = t_idx
                        option_logits_logged = True
                        break
                
                # If decision step found, extract logits AT THAT STEP
                if decision_step >= 0 and sample_logprobs and len(sample_logprobs) > decision_step:
                    step_lp = sample_logprobs[decision_step]
                    if step_lp:
                         # Prefer space-prefixed tokens if available in tokenizer
                         # We use the SAME logic as before to get IDs
                         # But we ADD BIAS manually because vLLM logprobs are pre-bias (raw model logits)
                         
                         for char in ['A', 'B', 'C', 'D']:
                             # Determine which token ID to look for: prefer space-prefixed
                             tid_target = option_ids_sp.get(char)
                             if tid_target is None:
                                 tid_target = option_ids.get(char)
                             
                             if tid_target is not None and tid_target in step_lp:
                                 raw_logit = step_lp[tid_target].logprob
                                 # ADD BIAS manually
                                 # logit_bias is dict: token_id -> bias_value
                                 bias_val = logit_bias.get(tid_target, 0.0)
                                 logit_dec_map[char] = raw_logit + bias_val
                
                # Assign to variables for saving (using _dec suffix)
                logit_A_dec = logit_dec_map['A'] if logit_dec_map['A'] > -1e9 else None
                logit_B_dec = logit_dec_map['B'] if logit_dec_map['B'] > -1e9 else None
                logit_C_dec = logit_dec_map['C'] if logit_dec_map['C'] > -1e9 else None
                logit_D_dec = logit_dec_map['D'] if logit_dec_map['D'] > -1e9 else None
                
                # Keep old sp logits for backward compatibility if needed, or just remove?
                # User said: "Keep existing columns; only add new ones."
                # So we keep the code that extracts step 0 logits (logit_A_sp etc) roughly or just let them be?
                # The user code I am replacing was extracting step 0.
                # I should PROBABLY keep step 0 extraction if I want to "Keep existing columns".
                # But the user said "log option logits at the ACTUAL ANSWER DECISION step t*, not always at step 1."
                # and "Save: decision_step=t*, logit_A_dec...".
                # I will KEEP the old step 0 extraction logic separately or just reuse variables?
                # The prompt says "Keep existing columns; only add new ones."
                # So I must RETAIN logit_A, logit_A_sp etc from step 0.
                
                # Restore Step 0 Logic (renamed to avoid conflict)
                logit_map_step0 = {'A': -np.inf, 'B': -np.inf, 'C': -np.inf, 'D': -np.inf}
                logit_map_sp_step0 = {'A': -np.inf, 'B': -np.inf, 'C': -np.inf, 'D': -np.inf}
                
                if sample_logprobs and len(sample_logprobs) > 0:
                     first_step_lp = sample_logprobs[0]
                     if first_step_lp:
                         for char, tid in option_ids.items():
                             if tid is not None and tid in first_step_lp:
                                 logit_map_step0[char] = first_step_lp[tid].logprob
                         for char, tid in option_ids_sp.items():
                             if tid is not None and tid in first_step_lp:
                                 logit_map_sp_step0[char] = first_step_lp[tid].logprob
                
                # Step 0 vars
                logit_A = logit_map_step0['A'] if logit_map_step0['A'] > -1e9 else None
                logit_B = logit_map_step0['B'] if logit_map_step0['B'] > -1e9 else None
                logit_C = logit_map_step0['C'] if logit_map_step0['C'] > -1e9 else None
                logit_D = logit_map_step0['D'] if logit_map_step0['D'] > -1e9 else None
                
                logit_A_sp = logit_map_sp_step0['A'] if logit_map_sp_step0['A'] > -1e9 else None
                logit_B_sp = logit_map_sp_step0['B'] if logit_map_sp_step0['B'] > -1e9 else None
                logit_C_sp = logit_map_sp_step0['C'] if logit_map_sp_step0['C'] > -1e9 else None
                logit_D_sp = logit_map_sp_step0['D'] if logit_map_sp_step0['D'] > -1e9 else None
                
                # ... row construction ...


                # Compute metrics locally (Task 3 logic, also embedded here for convenience/backup)
                # But strict Task 3 requires post-process script.
                # We will save these logits to CSV so post-process script works.
                
                row = {
                    'model': args.model_name,
                    'gamma': gamma,
                    'dataset': d_name,
                    'id': item['id'],
                    'y_correct': pred_correct,
                    'y_demand': y_demand,
                    'f1': feats[0],
                    'f2': feats[1],
                    'f3': feats[2], # Entropy
                    'f4': feats[3], # Margin
                    'f1_rand': feats[4],
                    'f2_rand': feats[5],
                    'marker_hit': marker_hit,
                    'avg_proj': feats[0],
                    'refusal_detected': refusal_detected,
                    'format_violation': format_violation,
                    'choice_margin': choice_margin,
                    'answer_entropy': answer_entropy,
                    # We will add logit_A, logit_B, logit_C, logit_D below
                    'option_logits_logged': option_logits_logged,
                    'decision_step': decision_step
                }
                if logit_A is not None: row['logit_A'] = logit_A
                if logit_B is not None: row['logit_B'] = logit_B
                if logit_C is not None: row['logit_C'] = logit_C
                if logit_D is not None: row['logit_D'] = logit_D
                
                if logit_A_sp is not None: row['logit_A_sp'] = logit_A_sp
                if logit_B_sp is not None: row['logit_B_sp'] = logit_B_sp
                if logit_C_sp is not None: row['logit_C_sp'] = logit_C_sp
                if logit_D_sp is not None: row['logit_D_sp'] = logit_D_sp
                
                if logit_A_dec is not None: row['logit_A_dec'] = logit_A_dec
                if logit_B_dec is not None: row['logit_B_dec'] = logit_B_dec
                if logit_C_dec is not None: row['logit_C_dec'] = logit_C_dec
                if logit_D_dec is not None: row['logit_D_dec'] = logit_D_dec
                
                results.append(row)
    
    # Save Raw Results
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"exp6_results_{args.direction_type}_{args.model_name}.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw results to {raw_path}")

    
    # Train Classifiers & Report Metrics
    summary_rows = []
    
    # Features groups
    feat_groups = {
        'Baseline': ['f3', 'f4'],
        'Real-U': ['f1', 'f2', 'f3', 'f4'],
        'Rand-U': ['f1_rand', 'f2_rand', 'f3', 'f4']
    }
    
    for gamma in gamma_list:
        for d_name in datasets.keys():
            subset = df[(df['gamma'] == gamma) & (df['dataset'] == d_name)]
            if len(subset) < 10:
                continue
            
            # Target 5A: y_correct
            y = subset['y_correct'].values
            if len(set(y)) < 2:
                # Can't train if only one class
                auc_scores = {k: 0.5 for k in feat_groups}
                acc_scores = {k: 0.5 for k in feat_groups}
            else:
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for group_name, cols in feat_groups.items():
                    X = subset[cols].values
                    aucs = []
                    accs = []
                    
                    try:
                        for train_idx, test_idx in kf.split(X, y):
                            clf = LogisticRegression(solver='liblinear')
                            clf.fit(X[train_idx], y[train_idx])
                            probs = clf.predict_proba(X[test_idx])[:, 1]
                            preds = clf.predict(X[test_idx])
                            
                            try:
                                auc = roc_auc_score(y[test_idx], probs)
                                aucs.append(auc)
                            except:
                                aucs.append(0.5)
                            accs.append(accuracy_score(y[test_idx], preds))
                        
                        avg_auc = np.mean(aucs)
                        avg_acc = np.mean(accs)
                    except:
                        avg_auc = 0.5
                        avg_acc = 0.5
                        
                    summary_rows.append({
                        'model': args.model_name,
                        'gamma': gamma,
                        'dataset': d_name,
                        'task': '5A_Correctness',
                        'feature_group': group_name,
                        'auc': avg_auc,
                        'accuracy': avg_acc
                    })

            # Target 5B: y_demand
            y_dem = subset['y_demand'].values
            if len(set(y_dem)) < 2:
                pass
            else:
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for group_name, cols in feat_groups.items():
                    X = subset[cols].values
                    aucs = []
                    try:
                        for train_idx, test_idx in kf.split(X, y_dem):
                            clf = LogisticRegression(solver='liblinear')
                            clf.fit(X[train_idx], y_dem[train_idx])
                            probs = clf.predict_proba(X[test_idx])[:, 1]
                            try:
                                auc = roc_auc_score(y_dem[test_idx], probs)
                                aucs.append(auc)
                            except:
                                pass
                        if aucs:
                            avg_auc = np.mean(aucs)
                        else:
                            avg_auc = 0.5
                    except:
                        avg_auc = 0.5
                        
                    summary_rows.append({
                        'model': args.model_name,
                        'gamma': gamma,
                        'dataset': d_name,
                        'task': '5B_ReasoningDemand',
                        'feature_group': group_name,
                        'auc': avg_auc,
                        'accuracy': 0 # Not focus
                    })

    # Save Summary
    summ_df = pd.DataFrame(summary_rows)
    summ_path = os.path.join(output_dir, f"exp6_summary_{args.direction_type}_{args.model_name}.csv")
    summ_df.to_csv(summ_path, index=False)
    print(f"Saved summary to {summ_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--direction_type", type=str, required=True, choices=['real', 'rand'])
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    run_experiment(args)

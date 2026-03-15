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
from utils import build_marker_token_ids, get_logit_bias_for_markers

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
        
        formatted.append({
            'prompt': prompt,
            'gold': ans,
            'type': dataset_type,
            'id': item.get('id', ''),
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
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)
    
    model_path = args.model_path
    dataset_N = config['dataset']['N']
    mmlu_path = config['dataset']['mmlu_path']
    medqa_path = config['dataset']['medqa_path']
    markers = config.get('markers', config.get('dataset', {}).get('markers', []))
    gamma_list = config['gamma_grid']
    logprobs_k = min(int(config.get('logprobs_k', 50)), 20)
    
    # Initialize LLM
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Build Marker IDs and U vectors
    marker_ids = build_marker_token_ids(model_path, markers)
    u_real, u_rand = build_u_vectors(marker_ids, vocab_size, seed=42)
    
    # Load Datasets
    datasets = {
        'mmlu': load_mcq_data(mmlu_path, 'mmlu', dataset_N),
        'medqa': load_mcq_data(medqa_path, 'medqa', dataset_N)
    }
    
    results = []
    
    for gamma in gamma_list:
        print(f"Running gamma={gamma}")
        
        logit_bias = get_logit_bias_for_markers(model_path, markers, gamma)
        
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
                # Simple check: Does output contain the gold letter?
                # E.g. "A" or "The answer is A"
                pred_correct = 0
                if gold.lower() in output_text.lower():
                     pred_correct = 1
                
                # Extract Features
                sample_logprobs = outputs[0].outputs[0].logprobs
                feats = extract_features_from_logprobs(sample_logprobs, u_real, u_rand)
                # [f1, f2, f3, f4, f1_rand, f2_rand]
                
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
                    'marker_hit': 0 # Answer-only usually has no markers, assumed 0 for this probe
                }
                results.append(row)
    
    # Save Raw Results
    df = pd.DataFrame(results)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, f"marker_irrelevant_label_results_{args.model_name}.csv")
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
    summ_path = os.path.join(output_dir, f"marker_irrelevant_summary_{args.model_name}.csv")
    summ_df.to_csv(summ_path, index=False)
    print(f"Saved summary to {summ_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)

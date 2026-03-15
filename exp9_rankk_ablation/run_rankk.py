import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_mcq_data(path, dataset_type, N, seed=42):
    # Same as build_Uk.py
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    import random
    random.seed(seed)
    random.shuffle(data)
    data = data[:N]
    
    formatted = []
    for idx, item in enumerate(data):
        q = item.get('question', '')
        choices = item.get('choices', [])
        ans = item.get('answer', '')
        
        options_text = ""
        labels = ['A', 'B', 'C', 'D', 'E']
        if isinstance(choices, list):
            for i, c in enumerate(choices):
                if i < len(labels):
                    options_text += f"{labels[i]}. {c}\n"
        
        prompt = f"Question: {q}\n{options_text}Answer ONLY with the correct option letter (A, B, C, D). Do not explain.\nAnswer:"
        
        formatted.append({
            'prompt': prompt,
            'gold': ans,
            'type': dataset_type,
            'id': item.get('id', idx),
        })
    return formatted

def _chunked(seq, chunk_size):
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]

def _softmax_1d(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def run_rankk(args):
    import yaml
    exp6_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments_markerfree",
        "exp6_correctness_defined_u",
    )
    config_path = os.path.join(exp6_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_path = args.model_path
    dataset_N = int(args.N)
    mmlu_path = config['dataset']['mmlu_path']
    medqa_path = config['dataset']['medqa_path']
    
    print(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
        max_logprobs=200000,
    )
    tokenizer = llm.get_tokenizer()
    vocab_size = int(tokenizer.vocab_size)
    
    datasets = {
        'mmlu': load_mcq_data(mmlu_path, 'mmlu', dataset_N, seed=args.seed),
        'medqa': load_mcq_data(medqa_path, 'medqa', dataset_N, seed=args.seed)
    }
    
    option_chars = ["A", "B", "C", "D"]
    option_ids = {}
    option_ids_sp = {}
    for char in option_chars:
        t1 = tokenizer.encode(char, add_special_tokens=False)
        option_ids[char] = t1[-1] if t1 else None
        t2 = tokenizer.encode(" " + char, add_special_tokens=False)
        option_ids_sp[char] = t2[-1] if t2 else None
        
    k_list = [int(x) for x in args.k_list.split(",")]
    gamma_grid = [float(x) for x in args.gamma_grid.split(",")]
    
    uk_by_k = {}
    for k in k_list:
        if "{k}" in args.Uk_path:
            uk_path = args.Uk_path.format(k=k)
        else:
            uk_path = args.Uk_path
        if not os.path.exists(uk_path):
            raise FileNotFoundError(f"Uk_path not found: {uk_path}")
        loaded = np.load(uk_path)
        Uk = np.asarray(loaded["Uk"], dtype=np.float32)
        if Uk.shape[0] != vocab_size or Uk.shape[1] != k:
            raise ValueError(f"Uk has shape {Uk.shape}, expected {(vocab_size, k)}")
        uk_by_k[k] = Uk

    totals = {}
    for k in k_list:
        for gamma in gamma_grid:
            condition = f"proj_k{k}"
            totals[(k, gamma, condition)] = {
                "n": 0,
                "acc_sum": 0.0,
                "entropy_sum": 0.0,
                "margin_sum": 0.0,
            }

    base_totals = {
        "n": 0,
        "acc_sum": 0.0,
        "entropy_sum": 0.0,
        "margin_sum": 0.0,
    }

    if args.normalize:
        print("Using Normalized Rank-K Projection: z' = z - (gamma / k) * U (U^T z)")
    else:
        print("Using Standard Rank-K Projection: z' = z - gamma * U (U^T z)")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=vocab_size,
        stop=["\n", "Question:", "User:"],
    )

    all_items = []
    for d_name, d_data in datasets.items():
        for item in d_data:
            all_items.append(item)

    batch_size = int(args.batch_size)
    total_batches = (len(all_items) + batch_size - 1) // batch_size
    for batch in tqdm(_chunked(all_items, batch_size), total=total_batches, desc="Batches"):
        prompts = [x["prompt"] for x in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for out, item in zip(outputs, batch):
            gold = str(item["gold"]).strip().upper()
            step_lp = None
            if out.outputs and out.outputs[0].logprobs:
                step_lp = out.outputs[0].logprobs[0]
            if not step_lp:
                continue

            z = np.zeros((vocab_size,), dtype=np.float32)
            for tid, val in step_lp.items():
                if 0 <= int(tid) < vocab_size:
                    z[int(tid)] = float(val.logprob)

            option_token_ids = []
            option_logits_base = []
            for c in option_chars:
                tid_sp = option_ids_sp.get(c)
                tid = tid_sp if tid_sp is not None else option_ids.get(c)
                if tid is None:
                    option_token_ids.append(None)
                    option_logits_base.append(np.nan)
                else:
                    option_token_ids.append(int(tid))
                    option_logits_base.append(float(z[int(tid)]))

            if any(tid is None for tid in option_token_ids):
                continue

            if any(np.isnan(option_logits_base)):
                continue

            gold_letter = gold[:1] if gold else ""

            pred_idx_base = int(np.argmax(option_logits_base))
            pred_letter_base = option_chars[pred_idx_base]
            acc_base = 1.0 if pred_letter_base == gold_letter else 0.0
            probs_base = _softmax_1d(option_logits_base)
            entropy_base = float(-np.sum(probs_base * np.log(probs_base + 1e-12)))
            top2_base = np.sort(np.asarray(option_logits_base))[-2:]
            margin_base = float(top2_base[-1] - top2_base[-2])

            base_totals["n"] += 1
            base_totals["acc_sum"] += acc_base
            base_totals["entropy_sum"] += entropy_base
            base_totals["margin_sum"] += margin_base

            for k, Uk in uk_by_k.items():
                coeffs = z @ Uk
                # Apply normalization if requested
                norm_factor = 1.0
                if args.normalize and k > 0:
                    norm_factor = 1.0 / k
                
                for gamma in gamma_grid:
                    effective_gamma = gamma * norm_factor
                    
                    proj_logits = []
                    for tid, base_logit in zip(option_token_ids, option_logits_base):
                        u_row = Uk[int(tid), :]
                        proj_logits.append(float(base_logit - effective_gamma * float(u_row @ coeffs)))

                    pred_idx = int(np.argmax(proj_logits))
                    pred_letter = option_chars[pred_idx]
                    acc = 1.0 if pred_letter == gold_letter else 0.0
                    probs = _softmax_1d(proj_logits)
                    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                    top2 = np.sort(np.asarray(proj_logits))[-2:]
                    margin = float(top2[-1] - top2[-2])

                    key = (k, gamma, f"proj_k{k}")
                    totals[key]["n"] += 1
                    totals[key]["acc_sum"] += acc
                    totals[key]["entropy_sum"] += entropy
                    totals[key]["margin_sum"] += margin

    rows = []
    if base_totals["n"] > 0:
        n_base = base_totals["n"]
        rows.append(
            {
                "model": args.model_name,
                "seed": int(args.seed),
                "N": int(args.N),
                "gamma": 0.0,
                "k": 0,
                "condition": "base",
                "accuracy_mean": base_totals["acc_sum"] / n_base,
                "entropy_mean": base_totals["entropy_sum"] / n_base,
                "margin_mean": base_totals["margin_sum"] / n_base,
                "strict_fail_rate": np.nan,
                "format_violation_rate": np.nan,
                "refusal_rate": np.nan,
                "marker_hit_rate": np.nan,
            }
        )
    for (k, gamma, condition), agg in totals.items():
        n = agg["n"]
        rows.append(
            {
                "model": args.model_name,
                "seed": int(args.seed),
                "N": int(args.N),
                "gamma": float(gamma),
                "k": int(k),
                "condition": condition,
                "accuracy_mean": (agg["acc_sum"] / n) if n else np.nan,
                "entropy_mean": (agg["entropy_sum"] / n) if n else np.nan,
                "margin_mean": (agg["margin_sum"] / n) if n else np.nan,
                "strict_fail_rate": np.nan,
                "format_violation_rate": np.nan,
                "refusal_rate": np.nan,
                "marker_hit_rate": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["condition", "k", "gamma"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved results to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="DeepSeek")
    parser.add_argument("--k_list", type=str, default="1,4")
    parser.add_argument("--gamma_grid", type=str, default="0,2,5,10")
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--Uk_path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--normalize", action="store_true", help="Normalize projection by 1/k")
    args = parser.parse_args()
    
    run_rankk(args)

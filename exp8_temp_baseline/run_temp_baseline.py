import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_exp6_config() -> Dict[str, Any]:
    import yaml

    cfg_path = "/root/workspace/experiments_markerfree/exp6_correctness_defined_u/config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_mcq_datasets(N: int, seed: int) -> List[Dict[str, Any]]:
    # Reuse Exp6 MCQ loader via direct path import (no package refactor)
    sys.path.append("/root/workspace/experiments_markerfree/exp6_correctness_defined_u")
    import run_marker_irrelevant_label as exp6_mod

    cfg = load_exp6_config()
    mmlu_path = cfg["dataset"]["mmlu_path"]
    medqa_path = cfg["dataset"]["medqa_path"]

    mmlu = exp6_mod.load_mcq_data(mmlu_path, "mmlu", N, seed)
    medqa = exp6_mod.load_mcq_data(medqa_path, "medqa", N, seed)

    all_samples = []
    all_samples.extend(mmlu)
    all_samples.extend(medqa)
    return all_samples


def build_option_token_ids(tokenizer) -> Tuple[Dict[str, int], Dict[str, int]]:
    option_chars = ["A", "B", "C", "D"]
    option_ids = {}
    option_ids_sp = {}
    for char in option_chars:
        t1 = tokenizer.encode(char, add_special_tokens=False)
        option_ids[char] = t1[-1] if t1 else None
        t2 = tokenizer.encode(" " + char, add_special_tokens=False)
        option_ids_sp[char] = t2[-1] if t2 else None
    return option_ids, option_ids_sp


def evaluate_condition(
    llm,
    tokenizer,
    samples: List[Dict[str, Any]],
    markers: List[str],
    option_ids: Dict[str, int],
    option_ids_sp: Dict[str, int],
    gamma: float,
    temperature: float,
    u_vector: np.ndarray | None,
    match_target: str,
    log_path: str | None = None,
) -> Dict[str, float]:
    sys.path.append("/root/workspace/experiments_markerfree/exp6_correctness_defined_u")
    import run_marker_irrelevant_label as exp6_mod

    if u_vector is not None:
        logit_bias = exp6_mod.vector_to_logit_bias(u_vector, gamma, scale=5.0)
    else:
        logit_bias = {}

    prompts = [s["prompt"] for s in samples]
    golds = [s["gold"] for s in samples]

    sp = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=10,
        logit_bias=logit_bias,
        logprobs=500,
        flat_logprobs=False,
        stop=["\n", "Question:", "User:"],
    )

    outputs = llm.generate(prompts, sp, use_tqdm=False)

    total = len(samples)
    correct_sum = 0.0
    marker_hit_sum = 0.0
    strict_fail_sum = 0.0

    entropy_vals: List[float] = []
    margin_vals: List[float] = []

    refusal_patterns = [
        "I can't",
        "I cannot",
        "I'm sorry",
        "I am sorry",
        "As an AI",
    ]

    log_rows = []

    for idx, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        gold = str(golds[idx]).strip()

        pred_correct = 1 if gold.lower() in text.lower() else 0
        correct_sum += pred_correct

        marker_hit = 0
        for m in markers:
            if m in text:
                marker_hit = 1
                break
        marker_hit_sum += marker_hit

        refusal_detected = 0
        for p in refusal_patterns:
            if p.lower() in text.lower():
                refusal_detected = 1
                break

        ans_chars = ["A", "B", "C", "D"]
        found_ans = any(c in text.upper() for c in ans_chars)
        format_violation = 0
        if not found_ans or len(text) > 100:
            format_violation = 1

        strict_fail = 1 if refusal_detected or format_violation else 0
        strict_fail_sum += strict_fail

        sample_logprobs = out.outputs[0].logprobs
        token_ids = out.outputs[0].token_ids

        decision_step = -1
        option_logits: Dict[str, float] = {"A": None, "B": None, "C": None, "D": None}

        for t_idx, tid in enumerate(token_ids):
            token_text = tokenizer.decode([tid], skip_special_tokens=True)
            import re

            match = re.match(r"^\s*\(?([ABCD])\)?\.?$", token_text)
            if match:
                decision_step = t_idx
                break

        if decision_step >= 0 and sample_logprobs and len(sample_logprobs) > decision_step:
            step_lp = sample_logprobs[decision_step]
            if step_lp:
                for ch in ["A", "B", "C", "D"]:
                    tid_target = option_ids_sp.get(ch)
                    if tid_target is None:
                        tid_target = option_ids.get(ch)
                    if tid_target is not None and tid_target in step_lp:
                        raw_logit = step_lp[tid_target].logprob
                        bias_val = logit_bias.get(int(tid_target), 0.0)
                        option_logits[ch] = raw_logit + bias_val

        logits_list = [v for v in option_logits.values() if v is not None]
        if len(logits_list) >= 2:
            arr = np.array(logits_list, dtype=np.float32)
            max_logit = np.max(arr)
            probs = np.exp(arr - max_logit)
            probs = probs / np.sum(probs)
            sorted_probs = np.sort(probs)
            margin = float(sorted_probs[-1] - sorted_probs[-2])
            entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
            margin_vals.append(margin)
            entropy_vals.append(entropy)

            if log_path is not None:
                sample = samples[idx]
                dataset = str(sample.get("type", ""))
                sid = str(sample.get("id", idx))
                log_rows.append(
                    {
                        "dataset": dataset,
                        "id": sid,
                        "gold": gold,
                        "temperature": temperature,
                        "y_correct": pred_correct,
                        "marker_hit": marker_hit,
                        "refusal_detected": refusal_detected,
                        "format_violation": format_violation,
                        "choice_margin": margin,
                        "answer_entropy": entropy,
                        "logit_A": option_logits["A"],
                        "logit_B": option_logits["B"],
                        "logit_C": option_logits["C"],
                        "logit_D": option_logits["D"],
                    }
                )

    accuracy_mean = correct_sum / total if total > 0 else 0.0
    marker_hit_rate = marker_hit_sum / total if total > 0 else 0.0
    strict_fail_rate = strict_fail_sum / total if total > 0 else 0.0

    entropy_mean = float(np.mean(entropy_vals)) if entropy_vals else float("nan")
    margin_mean = float(np.mean(margin_vals)) if margin_vals else float("nan")

    stat_for_match = entropy_mean if match_target == "entropy" else margin_mean

    if log_path is not None and log_rows:
        import csv

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "id",
                    "gold",
                    "temperature",
                    "y_correct",
                    "marker_hit",
                    "refusal_detected",
                    "format_violation",
                    "choice_margin",
                    "answer_entropy",
                    "logit_A",
                    "logit_B",
                    "logit_C",
                    "logit_D",
                ],
            )
            if write_header:
                writer.writeheader()
            for row in log_rows:
                writer.writerow(row)

    return {
        "accuracy_mean": accuracy_mean,
        "marker_hit_rate": marker_hit_rate,
        "strict_fail_rate": strict_fail_rate,
        "entropy_mean": entropy_mean,
        "margin_mean": margin_mean,
        "match_stat": stat_for_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model path")
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma_grid", type=str, default="0,10,25,40")
    parser.add_argument(
        "--temp_grid",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0",
    )
    parser.add_argument(
        "--match_target",
        type=str,
        choices=["entropy", "margin"],
        default="entropy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: outputs/tempbaseline_<modelname>.csv)",
    )
    args = parser.parse_args()

    model_path = args.model
    model_name = os.path.basename(os.path.normpath(model_path))

    gamma_list = parse_float_list(args.gamma_grid)
    temp_list = parse_float_list(args.temp_grid)

    cfg = load_exp6_config()
    markers = cfg.get("markers", [])
    default_temp = float(cfg["decoding"]["temperature"])

    print(f"Loading MCQ datasets (N={args.N}, seed={args.seed})...")
    samples = load_mcq_datasets(args.N, args.seed)

    print(f"Loading model from {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        logprobs_mode="raw_logits",
        max_logprobs=500,
    )
    tokenizer = llm.get_tokenizer()
    option_ids, option_ids_sp = build_option_token_ids(tokenizer)

    sys.path.append("/root/workspace/experiments_markerfree/exp6_correctness_defined_u")
    import run_marker_irrelevant_label as exp6_mod

    u_dir = "/root/workspace/experiments_markerfree/outputs/exp6_3"
    u_corr, _ = exp6_mod.load_u_vectors(model_name, u_dir)

    base_logits_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        f"exp8_base_logits_{model_name}.csv",
    )

    print("Evaluating base condition at default temperature T0...")
    base_default = evaluate_condition(
        llm,
        tokenizer,
        samples,
        markers,
        option_ids,
        option_ids_sp,
        gamma=0.0,
        temperature=default_temp,
        u_vector=None,
        match_target=args.match_target,
        log_path=base_logits_path,
    )

    base_cache: Dict[float, Dict[str, float]] = {}
    for T in temp_list:
        print(f"Evaluating base condition at temperature T={T}...")
        stats = evaluate_condition(
            llm,
            tokenizer,
            samples,
            markers,
            option_ids,
            option_ids_sp,
            gamma=0.0,
            temperature=T,
            u_vector=None,
            match_target=args.match_target,
            log_path=None,
        )
        base_cache[T] = stats

    proj_stats: Dict[float, Dict[str, float]] = {}
    for g in gamma_list:
        print(f"Evaluating projection condition at gamma={g}, T0={default_temp}...")
        stats = evaluate_condition(
            llm,
            tokenizer,
            samples,
            markers,
            option_ids,
            option_ids_sp,
            gamma=g,
            temperature=default_temp,
            u_vector=u_corr,
            match_target=args.match_target,
        )
        proj_stats[g] = stats

    match_key = "entropy_mean" if args.match_target == "entropy" else "margin_mean"
    deltas = [
        abs(proj_stats[g][match_key] - base_default[match_key]) for g in gamma_list
    ]
    if max(deltas) <= 1e-6:
        print(
            f"[Warning] Projection did not change {args.match_target} relative to base T0 for any gamma."
        )
    else:
        print(
            f"[Sanity] Projection changed {args.match_target} for at least one gamma; max Δ={max(deltas):.4f}"
        )

    rows = []
    for g in gamma_list:
        proj_val = proj_stats[g][match_key]

        best_T = None
        best_err = None
        best_stats = None
        for T in temp_list:
            stat_T = base_cache[T][match_key]
            err = abs(stat_T - proj_val)
            if best_err is None or err < best_err:
                best_err = err
                best_T = T
                best_stats = base_cache[T]

        print(
            f"[Match] gamma={g}: T*={best_T} proj_{args.match_target}={proj_val:.4f}, "
            f"base_{args.match_target}(T*)={best_stats[match_key]:.4f}, |Δ|={best_err:.4f}"
        )

        rows.append(
            {
                "model": model_name,
                "seed": args.seed,
                "N": args.N,
                "gamma": g,
                "condition": "base",
                "temperature": default_temp,
                "accuracy_mean": base_default["accuracy_mean"],
                "entropy_mean": base_default["entropy_mean"],
                "margin_mean": base_default["margin_mean"],
                "strict_fail_rate": base_default["strict_fail_rate"],
                "marker_hit_rate": base_default["marker_hit_rate"],
                "match_target": args.match_target,
            }
        )

        rows.append(
            {
                "model": model_name,
                "seed": args.seed,
                "N": args.N,
                "gamma": g,
                "condition": "proj",
                "temperature": default_temp,
                "accuracy_mean": proj_stats[g]["accuracy_mean"],
                "entropy_mean": proj_stats[g]["entropy_mean"],
                "margin_mean": proj_stats[g]["margin_mean"],
                "strict_fail_rate": proj_stats[g]["strict_fail_rate"],
                "marker_hit_rate": proj_stats[g]["marker_hit_rate"],
                "match_target": args.match_target,
            }
        )

        rows.append(
            {
                "model": model_name,
                "seed": args.seed,
                "N": args.N,
                "gamma": g,
                "condition": "tempmatch",
                "temperature": best_T,
                "accuracy_mean": best_stats["accuracy_mean"],
                "entropy_mean": best_stats["entropy_mean"],
                "margin_mean": best_stats["margin_mean"],
                "strict_fail_rate": best_stats["strict_fail_rate"],
                "marker_hit_rate": best_stats["marker_hit_rate"],
                "match_target": args.match_target,
            }
        )

    for T, stats in base_cache.items():
        rows.append(
            {
                "model": model_name,
                "seed": args.seed,
                "N": args.N,
                "gamma": 0.0,
                "condition": "base_tempgrid",
                "temperature": T,
                "accuracy_mean": stats["accuracy_mean"],
                "entropy_mean": stats["entropy_mean"],
                "margin_mean": stats["margin_mean"],
                "strict_fail_rate": stats["strict_fail_rate"],
                "marker_hit_rate": stats["marker_hit_rate"],
                "match_target": args.match_target,
            }
        )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    out_path = args.out
    if out_path is None:
        safe_name = model_name.replace("/", "_")
        out_path = os.path.join(out_dir, f"tempbaseline_{safe_name}.csv")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved temperature baseline results to {out_path}")

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

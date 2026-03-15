import os
import json
import argparse
import random
import time
import re
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

import sys
sys.path.append(os.path.dirname(__file__))
from logits_intervention import build_reasoning_processor

STRICT_SYSTEM = (
    "System: You are a helpful assistant. PROHIBITION: Do NOT output any\n"
    "reasoning or <think> tags. Output ONLY the final answer.\n"
    "If any reasoning is shown, it is considered a failure."
)
STRICT_USER_PREFIX = "User: "
STRICT_ASSISTANT_PREFIX = "Assistant:"

# Leakage detection rules from submission.tex
MARKER_PATTERNS = [
    r"<\s*think\s*>", r"<\s*/\s*think\s*>",
    r"step\s+by\s+step", r"let's\s+think",
    r"\bStep\b", r"Step\s*1\s*:", r"\bThought\s*:"
]
LEAKAGE_REGEXES = [re.compile(pat, flags=re.IGNORECASE | re.DOTALL) for pat in MARKER_PATTERNS]

CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
QUOTE_BLOCK_RE = re.compile(r"(^|\n)\s*>.*(\n|$)")


def strip_code_and_quotes(text: str) -> str:
    # Remove markdown code fences and quote lines
    cleaned = CODE_FENCE_RE.sub("", text)
    cleaned = QUOTE_BLOCK_RE.sub("", cleaned)
    return cleaned


def has_leakage(text: str) -> bool:
    t = strip_code_and_quotes(text)
    for rx in LEAKAGE_REGEXES:
        if rx.search(t):
            return True
    return False


def strict_fail(text: str) -> bool:
    # Per paper: failure if any reasoning is shown under Strict template
    return has_leakage(text)


def build_prompts(prompts_data: List[Dict[str, Any]]) -> List[str]:
    prompts = []
    for item in prompts_data:
        user = item["prompt"]
        prompt_text = f"{STRICT_SYSTEM}\n{STRICT_USER_PREFIX}{user}\n{STRICT_ASSISTANT_PREFIX}"
        prompts.append(prompt_text)
    return prompts


def load_or_create_prompts(data_path: str, source_files: List[str], count_needed: int, seed: int) -> List[Dict[str, Any]]:
    if os.path.exists(data_path):
        data = []
        with open(data_path, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
        if len(data) >= count_needed:
            return data
    # fallback: build from source files
    buffer = []
    for fpath in source_files:
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        prompt = item.get("prompt")
                        if not prompt:
                            instr = item.get("instruction")
                            inp = item.get("input", "")
                            if instr:
                                prompt = (instr + ("\n" + inp if inp else "")).strip()
                        if prompt:
                            buffer.append({"id": item.get("id", f"auto_{len(buffer)}"), "prompt": prompt})
                    except:
                        continue
    random.seed(seed)
    random.shuffle(buffer)
    buffer = buffer[:count_needed]
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w") as f:
        for it in buffer:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return buffer


def save_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    leak_flags = [1 if has_leakage(r["output"]) else 0 for r in rows]
    strict_flags = [1 if strict_fail(r["output"]) else 0 for r in rows]
    return {
        "leakage_rate": float(np.mean(leak_flags)) if leak_flags else 0.0,
        "strict_fail_rate": float(np.mean(strict_flags)) if strict_flags else 0.0,
        "n": len(rows)
    }


def generate_with_gamma(llm: LLM, prompts: List[str], gamma: float, model_path: str, max_tokens: int, temperature: float, top_p: float, seed: int) -> List[str]:
    # Use Option B (engine-native logit_bias) for vLLM V1 compatibility
    proc_info = build_reasoning_processor(model_path, gamma, option="B")
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        logit_bias=proc_info.get("logit_bias"),
        detokenize=True,
        skip_special_tokens=False
    )
    outs = llm.generate(prompts, sp, use_tqdm=True)
    texts = [o.outputs[0].text for o in outs]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--mode", choices=["dryrun", "full"], required=True)
    parser.add_argument("--output_root", default="intervention_minimal/results")
    parser.add_argument("--fig_root", default="intervention_minimal/figures")
    parser.add_argument("--data_path", default="intervention_minimal/data/prompts_strict.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    # Prepare prompts
    if args.mode == "dryrun":
        N_main = 5
        N_sweep = 0
    else:
        N_main = 200
        N_sweep = 80

    source_files = [
        "/root/workspace/train/data/medical/finetune/valid_en_1.json",
        "/root/workspace/train/data/medical/finetune/valid_zh_0.json"
    ]
    total_needed = max(N_main, N_sweep)
    prompts_data = load_or_create_prompts(args.data_path, source_files, total_needed, args.seed)
    prompts_main = build_prompts(prompts_data[:N_main])
    prompts_sweep = build_prompts(prompts_data[:N_sweep]) if N_sweep > 0 else []

    os.makedirs(args.output_root, exist_ok=True)

    for model_path in args.models:
        model_id = os.path.basename(model_path.rstrip("/"))
        model_out_dir = os.path.join(args.output_root, model_id)
        os.makedirs(model_out_dir, exist_ok=True)

        print(f"Loading model {model_id}...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            seed=args.seed
        )

        # Baseline gamma=0 on N_main
        print(f"Running baseline gamma=0 for {model_id} (N={len(prompts_main)})")
        texts_baseline = generate_with_gamma(llm, prompts_main, gamma=0.0, model_path=model_path,
                                             max_tokens=args.max_tokens, temperature=args.temperature,
                                             top_p=args.top_p, seed=args.seed)
        rows_baseline = []
        for i, t in enumerate(texts_baseline):
            rows_baseline.append({
                "model": model_id, "prompt_id": prompts_data[i]["id"], "prompt": prompts_data[i]["prompt"],
                "output": t, "gamma": 0.0, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p
            })
        save_jsonl(os.path.join(model_out_dir, "baseline_gamma0_main.jsonl"), rows_baseline)
        metrics_baseline = compute_metrics(rows_baseline)
        with open(os.path.join(model_out_dir, "baseline_metrics.json"), "w") as f:
            json.dump(metrics_baseline, f, indent=2)

        # Intervention gamma=1.0 on N_main (paired)
        print(f"Running intervention gamma=1.0 for {model_id} (N={len(prompts_main)})")
        texts_intervention = generate_with_gamma(llm, prompts_main, gamma=1.0, model_path=model_path,
                                                 max_tokens=args.max_tokens, temperature=args.temperature,
                                                 top_p=args.top_p, seed=args.seed)
        rows_intervention = []
        for i, t in enumerate(texts_intervention):
            rows_intervention.append({
                "model": model_id, "prompt_id": prompts_data[i]["id"], "prompt": prompts_data[i]["prompt"],
                "output": t, "gamma": 1.0, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p
            })
        save_jsonl(os.path.join(model_out_dir, "intervention_gamma1_main.jsonl"), rows_intervention)
        metrics_intervention = compute_metrics(rows_intervention)
        with open(os.path.join(model_out_dir, "intervention_metrics.json"), "w") as f:
            json.dump(metrics_intervention, f, indent=2)

        # Sweep gamma (0.0 to 5.0) on N_sweep
        if N_sweep > 0:
            gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
            print(f"Running gamma sweep {gammas} for {model_id} (N={len(prompts_sweep)})")
            sweep_rows = []
            for g in gammas:
                texts_sw = generate_with_gamma(llm, prompts_sweep, gamma=g, model_path=model_path,
                                               max_tokens=args.max_tokens, temperature=args.temperature,
                                               top_p=args.top_p, seed=args.seed)
                for i, t in enumerate(texts_sw):
                    sweep_rows.append({
                        "model": model_id, "prompt_id": prompts_data[i]["id"], "prompt": prompts_data[i]["prompt"],
                        "output": t, "gamma": g, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p
                    })
            save_jsonl(os.path.join(model_out_dir, "sweep_gamma_all.jsonl"), sweep_rows)
            # Compute metrics per gamma
            sweep_metrics = {}
            for g in gammas:
                subset = [r for r in sweep_rows if r["gamma"] == g]
                sweep_metrics[str(g)] = compute_metrics(subset)
            with open(os.path.join(model_out_dir, "sweep_metrics.json"), "w") as f:
                json.dump(sweep_metrics, f, indent=2)

    print("Done.")



if __name__ == "__main__":
    main()

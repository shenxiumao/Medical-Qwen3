
import os
import json
import argparse
import random
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Constants & Templates ---

STRICT_SYSTEM = (
    "System: You are a helpful assistant. PROHIBITION: Do NOT output any\n"
    "reasoning or <think> tags. Output ONLY the final answer.\n"
    "If any reasoning is shown, it is considered a failure."
)
STRICT_USER_PREFIX = "User: "
STRICT_ASSISTANT_PREFIX = "Assistant:"

MARKER_PATTERNS = [
    r"<\s*think\s*>", r"<\s*/\s*think\s*>",
    r"step\s+by\s+step", r"let's\s+think",
    r"\bStep\b", r"Step\s*1\s*:", r"\bThought\s*:"
]
LEAKAGE_REGEXES = [re.compile(pat, flags=re.IGNORECASE | re.DOTALL) for pat in MARKER_PATTERNS]

CODE_FENCE_RE = re.compile(r"```.*?```", flags=re.DOTALL)
QUOTE_BLOCK_RE = re.compile(r"(^|\n)\s*>.*(\n|$)")

MODELS = {
    "Qwen2.5-7B-Instruct": "/root/workspace/model/Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct": "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Qwen-7B": "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

# --- Helpers ---

def strip_code_and_quotes(text: str) -> str:
    cleaned = CODE_FENCE_RE.sub("", text)
    cleaned = QUOTE_BLOCK_RE.sub("", cleaned)
    return cleaned

def has_leakage(text: str) -> bool:
    t = strip_code_and_quotes(text)
    for rx in LEAKAGE_REGEXES:
        if rx.search(t):
            return True
    return False

def build_marker_token_ids(model_path: str, markers: List[str]) -> List[int]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids: List[int] = []
    for m in markers:
        try:
            enc = tok.encode(m, add_special_tokens=False)
            ids.extend(enc)
        except Exception:
            continue
    ids = [int(x) for x in set(ids) if isinstance(x, int)]
    return ids

def build_ablation_logit_bias(model_path: str, gamma: float, mode: str) -> Dict[int, float]:
    """
    mode:
      "all": all markers (original)
      "think_only": only <think>, </think>
      "no_think": everything EXCEPT <think>, </think>
    """
    all_markers = [
        "<think>", "</think>", "Step", "Step 1:", "Thought:", "let's think", "step by step"
    ]
    
    if mode == "all":
        target_markers = all_markers
    elif mode == "think_only":
        target_markers = ["<think>", "</think>"]
    elif mode == "no_think":
        target_markers = [m for m in all_markers if "think" not in m]
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    marker_ids = build_marker_token_ids(model_path, target_markers)
    
    # Use Option B logic (scalar suppression)
    bias_scale = 5.0
    logit_bias = {int(tid): float(-gamma * bias_scale) for tid in marker_ids}
    return logit_bias

# --- Execution ---

def run_experiment(model_name: str, model_path: str, prompts: List[str], 
                   configs: List[Dict[str, Any]], output_dir: str):
    print(f"Loading {model_name} from {model_path}...")
    try:
        # TP=4 as requested
        llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True, 
                  gpu_memory_utilization=0.9, max_model_len=4096)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return

    results = []
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    for cfg in configs:
        gamma = cfg["gamma"]
        mode = cfg.get("mode", "all")
        desc = cfg["desc"]
        
        print(f"Running {desc} (gamma={gamma}, mode={mode})...")
        
        if gamma > 0:
            logit_bias = build_ablation_logit_bias(model_path, gamma, mode)
        else:
            logit_bias = None
            
        sp = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=512,
            logit_bias=logit_bias,
            detokenize=True
        )
        
        # Run inference
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        # Metrics
        leak_count = 0
        generated_texts = []
        for o in outputs:
            text = o.outputs[0].text
            generated_texts.append(text)
            if has_leakage(text):
                leak_count += 1
        
        leak_rate = leak_count / len(prompts)
        print(f"  -> Leakage Rate: {leak_rate:.2f}")
        
        results.append({
            "model": model_name,
            "gamma": gamma,
            "mode": mode,
            "desc": desc,
            "leakage_rate": leak_rate,
            "N": len(prompts)
        })
        
        # Save raw outputs to model subdirectory
        save_path = os.path.join(model_output_dir, f"{desc}.jsonl")
        with open(save_path, "w") as f:
            for i, text in enumerate(generated_texts):
                json.dump({
                    "id": i,
                    "model": model_name,
                    "gamma": gamma,
                    "mode": mode,
                    "output": text,
                    "leakage": has_leakage(text)
                }, f)
                f.write("\n")
                
    # Cleanup vLLM to free memory for next model
    # Note: Since we use subprocess isolation, strict cleanup is less critical here.
    # We avoid accessing internal attributes like model_executor which vary by vLLM version.
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
        
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="/root/workspace/intervention_robustness/results")
    parser.add_argument("--data_path", default="/root/workspace/intervention_minimal/data/prompts_strict.jsonl")
    parser.add_argument("--N", type=int, default=40)
    parser.add_argument("--model_alias", type=str, default=None, help="If set, run only this model. If None, run all via subprocess.")
    args = parser.parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    fig_dir = os.path.join(os.path.dirname(args.output_root), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # If master process (no model_alias), spawn workers
    if args.model_alias is None:
        print(f"Master process: Spawning workers for {len(MODELS)} models (TP=4)...")
        import subprocess
        import sys
        
        processes = []
        for m_alias in MODELS.keys():
            print(f"Starting worker for {m_alias}...")
            cmd = [sys.executable, __file__, 
                   "--model_alias", m_alias, 
                   "--output_root", args.output_root,
                   "--data_path", args.data_path,
                   "--N", str(args.N)]
            # Run sequentially to avoid GPU contention
            subprocess.check_call(cmd)
            
        print("All workers finished. Aggregating results and plotting...")
        aggregate_and_plot(args.output_root, fig_dir)
        return

    # --- Worker Process Logic ---
    
    # Load prompts
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    # Random sample
    random.seed(42)
    random.shuffle(data)
    subset = data[:args.N]
    
    prompts = [
        f"{STRICT_SYSTEM}\n{STRICT_USER_PREFIX}{item['prompt']}\n{STRICT_ASSISTANT_PREFIX}"
        for item in subset
    ]
    print(f"[{args.model_alias}] Loaded {len(prompts)} prompts.")
    
    # Define Configurations
    # 1. Extended Gamma Sweep
    configs_sanity = [
        {"gamma": 0.0, "desc": "sanity_gamma0.0"},
        {"gamma": 1.0, "desc": "sanity_gamma1.0"},
        {"gamma": 1.5, "desc": "sanity_gamma1.5"},
        {"gamma": 2.0, "desc": "sanity_gamma2.0"},
        {"gamma": 5.0, "desc": "sanity_gamma5.0"},
    ]
    
    # 2. Marker Ablation (at gamma=1.0)
    configs_ablation = [
        {"gamma": 1.0, "mode": "think_only", "desc": "ablation_think_only"},
        {"gamma": 1.0, "mode": "no_think", "desc": "ablation_no_think"},
        # "all" is already covered in sanity_gamma1.0
    ]
    
    all_configs = configs_sanity + configs_ablation
    
    m_path = MODELS[args.model_alias]
    res = run_experiment(args.model_alias, m_path, prompts, all_configs, args.output_root)
    
    # Save individual result for aggregation (inside model dir)
    if res:
        model_output_dir = os.path.join(args.output_root, args.model_alias)
        res_path = os.path.join(model_output_dir, "robustness_metrics.json")
        with open(res_path, "w") as f:
            json.dump(res, f, indent=2)

def aggregate_and_plot(output_root, fig_dir):
    all_results = []
    # Find all robustness_metrics.json in subdirectories
    for item in os.listdir(output_root):
        sub_dir = os.path.join(output_root, item)
        if os.path.isdir(sub_dir):
            metrics_file = os.path.join(sub_dir, "robustness_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    all_results.extend(json.load(f))
                
    if not all_results:
        print("No results found to plot.")
        return
        
    # Save aggregated metrics JSON
    with open(os.path.join(output_root, "robustness_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Save aggregated metrics CSV
    import csv
    if all_results:
        keys = all_results[0].keys()
        with open(os.path.join(output_root, "robustness_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        
    # --- Plotting ---
    plot_sanity(all_results, fig_dir)
    plot_ablation(all_results, fig_dir)


def plot_sanity(results, fig_dir):
    plt.figure(figsize=(10, 6))
    models = sorted(list(set(r["model"] for r in results)))
    
    for m in models:
        # Filter for sanity configs (mode="all")
        subset = [r for r in results if r["model"] == m and r.get("mode", "all") == "all"]
        subset.sort(key=lambda x: x["gamma"])
        
        gammas = [r["gamma"] for r in subset]
        leaks = [r["leakage_rate"] for r in subset]
        
        plt.plot(gammas, leaks, marker="o", label=m)
        
    plt.xlabel("Gamma")
    plt.ylabel("Leakage Rate")
    plt.title(f"Robustness: Gamma Monotonicity Check (N={results[0]['N']})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "figure_gamma_sanity.png"))
    plt.close()

def plot_ablation(results, fig_dir):
    plt.figure(figsize=(12, 6))
    models = sorted(list(set(r["model"] for r in results)))
    
    # We want to compare: Gamma=1.0 (All) vs Think-only vs No-think
    # Also include Baseline (Gamma=0) for reference?
    
    bar_width = 0.2
    x = np.arange(len(models))
    
    modes = ["all", "think_only", "no_think"]
    labels = ["All Markers", "Only <think>", "No <think> (Step, etc.)"]
    
    for i, (mode, label) in enumerate(zip(modes, labels)):
        vals = []
        for m in models:
            # Find the entry
            # Note: "all" might be under desc="sanity_gamma1.0"
            if mode == "all":
                entry = next((r for r in results if r["model"] == m and r["gamma"] == 1.0 and r.get("mode", "all") == "all"), None)
            else:
                entry = next((r for r in results if r["model"] == m and r["gamma"] == 1.0 and r.get("mode") == mode), None)
            
            vals.append(entry["leakage_rate"] if entry else 0.0)
            
        plt.bar(x + i*bar_width, vals, width=bar_width, label=label)
        
    plt.xticks(x + bar_width, models, rotation=15)
    plt.ylabel("Leakage Rate")
    plt.title(f"Robustness: Marker Ablation (Gamma=1.0, N={results[0]['N']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure_marker_ablation.png"))
    plt.close()

if __name__ == "__main__":
    main()

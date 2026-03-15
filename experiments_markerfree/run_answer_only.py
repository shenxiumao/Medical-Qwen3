import argparse
import os
import csv
import torch
from vllm import LLM, SamplingParams
from utils import load_config, load_prompts, format_prompt_answer_only, get_logit_bias_for_markers, has_leakage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_alias", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    model_path = config['models'][args.model_alias]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{args.model_alias}] Starting Answer-Only Experiment...")

    # Load data & Format with "Answer only" prompt
    raw_data = load_prompts(config['dataset']['path'], config['dataset']['N'], config['seed'])
    prompts = [format_prompt_answer_only(d['prompt']) for d in raw_data]

    # Init LLM
    try:
        llm = LLM(model=model_path, tensor_parallel_size=4, trust_remote_code=True,
                  gpu_memory_utilization=0.9, max_model_len=4096)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []
    markers = config['markers']

    for gamma in config['gamma_grid']:
        print(f"Running Gamma={gamma}...")
        
        logit_bias = get_logit_bias_for_markers(model_path, markers, gamma, scale=5.0)
        
        sp = SamplingParams(
            temperature=config['decoding']['temperature'],
            top_p=config['decoding']['top_p'],
            max_tokens=config['decoding']['max_tokens'],
            logit_bias=logit_bias,
            detokenize=True
        )
        
        outputs = llm.generate(prompts, sp, use_tqdm=True)
        
        leak_count = 0
        
        for o in outputs:
            text = o.outputs[0].text
            if has_leakage(text, custom_markers=markers):
                leak_count += 1
                
        leak_rate = leak_count / len(prompts)
        print(f"  Gamma={gamma}: Leakage={leak_rate:.2f}")
        
        results.append({
            "model": args.model_alias,
            "gamma": gamma,
            "leakage_rate": leak_rate,
            "experiment": "answer_only"
        })

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
    csv_path = os.path.join(output_dir, f"answer_only_{args.model_alias}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

if __name__ == "__main__":
    main()

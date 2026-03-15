from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pandas as pd
import json
import os
import glob
import matplotlib.pyplot as plt

def main():
    base_model_path = "/root/workspace/model/Qwen/Qwen3-14B"
    checkpoints_dir = "/root/workspace/experiments/subspace_aware_merge_toy/checkpoints"
    data_path = "/root/workspace/train/benchmark_data/mmlu_medical_test.jsonl"
    output_dir = "/root/workspace/experiments/subspace_aware_merge_toy"
    
    # Load data
    print(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_json(data_path, lines=True, orient='records')
    df = df.head(100) # N=100
    
    prompts = []
    
    for _, row in df.iterrows():
        q = row.get('question', '')
        opts = row.get('options', {})
        if isinstance(opts, dict):
            # {'A': ..., 'B': ...}
            prompt = f"{q}\nA. {opts.get('A','')}\nB. {opts.get('B','')}\nC. {opts.get('C','')}\nD. {opts.get('D','')}\nAnswer:"
        elif isinstance(opts, list):
            prompt = f"{q}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:"
        else:
            prompt = f"{q}\nAnswer:"
        prompts.append(prompt)
        
    # Initialize LLM
    print("Initializing LLM...")
    # Using tensor_parallel_size=4 as per environment capability/Exp6 precedent
    llm = LLM(model=base_model_path, enable_lora=True, max_lora_rank=16, tensor_parallel_size=4, enforce_eager=True) 
    
    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    
    results = []
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints dir {checkpoints_dir} not found. Did you run merge script?")
        return

    adapters = sorted(glob.glob(os.path.join(checkpoints_dir, "*")))
    print(f"Found {len(adapters)} adapters.")
    
    for i, adapter_path in enumerate(adapters):
        adapter_name = os.path.basename(adapter_path)
        print(f"Evaluating {adapter_name} ({i+1}/{len(adapters)})...")
        
        # Define LoRA request
        lora_req = LoRARequest(adapter_name, i+1, adapter_path)
        
        try:
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        except Exception as e:
            print(f"Error generating for {adapter_name}: {e}")
            continue
        
        correct_count = 0
        leakage_count = 0
        
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Determine true answer
            ans_val = df.iloc[j].get('answer', '')
            if isinstance(ans_val, int):
                true_answer = ["A", "B", "C", "D"][ans_val]
            else:
                true_answer = str(ans_val)[0] # 'A'
            
            stripped = generated_text.strip()
            # Simple check: does it start with the letter?
            if stripped.startswith(true_answer):
                correct_count += 1
                
            # Leakage: Check for "The answer is"
            if "The answer is" in generated_text:
                leakage_count += 1
        
        acc = correct_count / len(prompts)
        leak = leakage_count / len(prompts)
        
        print(f"  Acc: {acc:.2f}, Leak: {leak:.2f}")
        
        # Parse adapter name
        try:
            parts = adapter_name.split('_')
            # merge_alpha0.5_gamma0_naive
            alpha = float(parts[1].replace("alpha", ""))
            gamma = float(parts[2].replace("gamma", ""))
            atype = "_".join(parts[3:]) 
        except:
            alpha, gamma, atype = 0, 0, "unknown"
        
        results.append({
            "adapter": adapter_name,
            "alpha": alpha,
            "gamma": gamma,
            "type": atype,
            "accuracy": acc,
            "leakage": leak
        })
        
    # Save CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "eval_summary.csv"), index=False)
    print(f"Saved summary to {os.path.join(output_dir, 'eval_summary.csv')}")
    
    # Plot
    try:
        plt.figure(figsize=(10, 6))
        for name, group in res_df.groupby("type"):
            plt.scatter(group["leakage"], group["accuracy"], label=name, alpha=0.7)
            
            for _, row in group.iterrows():
                plt.annotate(f"g{row['gamma']}", (row['leakage'], row['accuracy']), fontsize=8)
                
        plt.xlabel("Leakage Rate")
        plt.ylabel("Accuracy")
        plt.title("Toy Merge Evaluation")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "leakage_vs_accuracy.png"))
        print("Saved plot.")
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    main()

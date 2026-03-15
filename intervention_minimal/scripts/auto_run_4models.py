import os
import shutil
import subprocess
import sys

def main():
    # 1. Define models
    models = [
        "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct",
        "/root/workspace/model/Qwen/Qwen2.5-7B-Instruct",
        "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ]

    # Verify models exist
    for m in models:
        if not os.path.exists(m):
            print(f"Error: Model path not found: {m}")
            # Try to find it in other locations or list available models
            # But for now, just warn and continue or exit?
            # We'll exit to avoid wasting time.
            sys.exit(1)

    # 2. Cleanup generated data
    print("Cleaning up generated data...")
    dirs_to_clean = [
        "/root/workspace/intervention_minimal/results",
        "/root/workspace/intervention_minimal/figures",
        # "/root/workspace/intervention_minimal/data" # Keep prompts? User said "all generated data". 
        # prompts_strict.jsonl might be considered generated if created from source.
        # But load_or_create_prompts checks existence. 
        # I'll keep data/prompts_strict.jsonl to save time, unless user strictly wants fresh prompts.
        # "all generated data" usually implies outputs. 
        # But to be safe and "fully automated", I will let the script recreate prompts.
        "/root/workspace/intervention_minimal/data"
    ]
    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"Removing {d}...")
            shutil.rmtree(d)

    # 3. Run evaluation for each model
    script_path = "/root/workspace/intervention_minimal/scripts/run_vllm_eval.py"
    
    # Define absolute paths for outputs
    output_root = "/root/workspace/intervention_minimal/results"
    fig_root = "/root/workspace/intervention_minimal/figures"
    data_path = "/root/workspace/intervention_minimal/data/prompts_strict.jsonl"

    for model in models:
        print(f"\n{'='*50}")
        print(f"Running evaluation for {os.path.basename(model)}...")
        print(f"{'='*50}\n")
        
        cmd = [
            "python", script_path,
            "--models", model,
            "--mode", "full",
            "--max_tokens", "512",
            "--temperature", "0.6",
            "--output_root", output_root,
            "--fig_root", fig_root,
            "--data_path", data_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running model {model}: {e}")
            continue

    # 4. Generate figures
    print("\nGenerating figures...")
    fig_script = "/root/workspace/intervention_minimal/scripts/make_figures.py"
    if os.path.exists(fig_script):
        # Pass absolute paths to make_figures.py as well
        cmd_fig = [
            "python", fig_script,
            "--results_root", output_root,
            "--fig_root", fig_root
        ]
        subprocess.run(cmd_fig, check=False)
    else:
        print("Figure generation script not found.")

    print("\nAll done.")

if __name__ == "__main__":
    main()

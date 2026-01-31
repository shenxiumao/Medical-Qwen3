import os
import subprocess
import shutil
import queue
import json
import uuid
import datetime
from concurrent.futures import ThreadPoolExecutor

# Configs
BASE_MODEL = "/root/workspace/model/Qwen/Qwen3-14B"
MODELS_DIR = "/root/workspace/train/merged_models_v2"
OUTPUT_DIR = "/root/workspace/train/ablation_results_eval_v2"
DATASET_DIR = "/root/workspace/train/data"

# Ablation Settings (alpha_pt, alpha_sft)
WEIGHTS = [
    (0.0, 1.0),       # SFT Only
    (0.3, 0.7),       # Proposed
    (1.0, 0.0)        # PT Only
]

# Templates to test: (template_name, enable_thinking, output_suffix)
# "think" -> qwen3, enable_thinking=True
# "nothing" -> qwen3_nothink, enable_thinking=False
TEST_CONFIGS = [
    ("qwen3", False, "think_off"),   # 复现旧口径的锚点
    ("qwen3", True,  "think_on"),    # 显式开启 thinking
    ("qwen3_nothink", False, "nothink"),
]

# Decode Presets
PRESETS = [
    {
        "name": "PresetA",
        "desc": "Deterministic",
        "temperature": 0.0,  # Actually < 1e-5 usually, or handled by do_sample=False
        "top_p": 1.0,
        "do_sample": False,
        "seed": 42
    },
    {
        "name": "PresetB",
        "desc": "Stochastic",
        "temperature": 0.6,
        "top_p": 0.8,
        "do_sample": True,
        "seed": 42
    }
]

def run_cmd(cmd, env=None):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True, env=env)

def evaluate_single_task(task_args, gpu_id):
    pt_w, sft_w, tmpl, thinking, tmpl_suffix, preset = task_args
    preset_name = preset["name"]
    
    print(f"\n[GPU {gpu_id}] Processing Eval for PT={pt_w}, SFT={sft_w}, {tmpl_suffix}, {preset_name}...")
    
    # Environment for this worker (isolate GPU)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Disable SwanLab and WandB
    env["SWANLAB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    
    try:
        # 1. Prepare Model (Use Pre-merged)
        fingerprint = "unknown"
        if pt_w == "base":
            current_model_path = BASE_MODEL
            fingerprint = "base_model"
        else:
            current_model_path = f"{MODELS_DIR}/merged_pt{pt_w}_sft{sft_w}"
            if not os.path.exists(current_model_path):
                print(f"[GPU {gpu_id}] Model not found: {current_model_path}. Skipping.")
                return
            # Try to read merge_meta.json for fingerprint
            meta_path = os.path.join(current_model_path, "merge_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        m = json.load(f)
                        fingerprint = m.get("fingerprint", "unknown")
                except:
                    pass
        
        # 2. LLaMA-Factory Predict (Eval)
        print(f"[GPU {gpu_id}] Running LLaMA-Factory Predict for {tmpl_suffix} / {preset_name}...")
        
        # Construct output dir: eval_ptX_sftY_think_PresetA
        eval_output = f"{OUTPUT_DIR}/eval_pt{pt_w}_sft{sft_w}_{tmpl_suffix}_{preset_name}"
        
        # Write run_meta.json BEFORE running
        if not os.path.exists(eval_output):
            os.makedirs(eval_output, exist_ok=True)
        
        run_meta = {
            "model_path": current_model_path,
            "realpath": os.path.realpath(current_model_path),
            "template": tmpl,
            "enable_thinking": thinking,
            "temperature": preset["temperature"],
            "top_p": preset["top_p"],
            "do_sample": preset["do_sample"],
            "seed": preset["seed"],
            "fingerprint": fingerprint,
            "preset_name": preset_name,
            "desc": preset["desc"]
        }
        
        with open(os.path.join(eval_output, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

        sample_arg = f"--do_sample {str(preset['do_sample']).lower()}" 
        
        eval_cmd = (
            f"llamafactory-cli train "
            f"--stage sft "
            f"--model_name_or_path {current_model_path} "
            f"--dataset_dir {DATASET_DIR} "
            f"--eval_dataset medical_f5_valid,medical_f6_valid "
            f"--template {tmpl} "
            f"--finetuning_type lora "
            f"--output_dir {eval_output} "
            f"--overwrite_output_dir "
            f"--per_device_eval_batch_size 20 "
            f"--predict_with_generate "
            f"--do_predict "
            f"--top_p {preset['top_p']} "
            f"--temperature {preset['temperature']} "
            f"{sample_arg} "
            f"--seed {preset['seed']} "
            f"--cutoff_len 40960 "
            f"--max_new_tokens 4096 "
            f"--trust_remote_code True "
            f"--flash_attn fa2 "
            f"--ddp_timeout 180000000 "
            f"--enable_thinking {thinking} "
            f"--report_to none "
            f"--use_swanlab False "
        )
        
        try:
            run_cmd(eval_cmd, env=env)
        except Exception as e:
            print(f"[GPU {gpu_id}] Eval failed for {tmpl_suffix}/{preset_name}: {e}")

    except Exception as e:
        print(f"[GPU {gpu_id}] Error in evaluate_single_task: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Queue of tasks (Granular: Weight x Template x Preset)
    task_queue = queue.Queue()
    
    # Create all combinations
    for (pt_w, sft_w) in WEIGHTS:
        for (tmpl, thinking, tmpl_suffix) in TEST_CONFIGS:
            for preset in PRESETS:
                task = (pt_w, sft_w, tmpl, thinking, tmpl_suffix, preset)
                task_queue.put(task)

    total_tasks = task_queue.qsize()
    print(f"Total evaluation tasks: {total_tasks}")

    def worker(gpu_id):
        while not task_queue.empty():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break
            
            try:
                evaluate_single_task(task, gpu_id)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing task: {e}")
            finally:
                task_queue.task_done()

    # Launch 4 threads, one for each GPU
    num_gpus = 4
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(worker, i) for i in range(num_gpus)]
        
        # Wait for all
        for f in futures:
            f.result()

if __name__ == "__main__":
    main()

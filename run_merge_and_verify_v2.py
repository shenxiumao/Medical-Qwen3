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
PT_ADAPTER = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/train_pt_2025-12-17-10-58-51"
SFT_ADAPTER = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/sft_medical_2025-12-13-17-11-36"
MERGE_SCRIPT = "/root/workspace/train/merge_qwen_adapters_scaled.py"
VERIFY_SCRIPT = "/root/workspace/train/verify_merge.py"
MODELS_DIR = "/root/workspace/train/merged_models_v2"

# Ablation Settings (alpha_pt, alpha_sft)
WEIGHTS = [
    (0.0, 1.0),       # SFT Only
    (0.3, 0.7),       # Proposed
    (1.0, 0.0)        # PT Only
]

def run_cmd(cmd, env=None):
    print(f"Running: {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, env=env)

def merge_and_verify(pt_w, sft_w, gpu_id):
    if pt_w == "base":
        print(f"[GPU {gpu_id}] Skipping merge for Base model", flush=True)
        return

    print(f"\n[GPU {gpu_id}] Merging PT={pt_w}, SFT={sft_w}...", flush=True)
    
    # Define output directory for this config
    merged_dir = f"{MODELS_DIR}/merged_pt{pt_w}_sft{sft_w}"
    
    # Environment for this worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    
    if os.path.exists(merged_dir):
        meta_path = os.path.join(merged_dir, "merge_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("verification_status") == "success":
                    print(f"[GPU {gpu_id}] Output dir {merged_dir} exists and verified. Skipping.", flush=True)
                    return
            except Exception:
                pass
        
        print(f"[GPU {gpu_id}] Output dir exists but incomplete/unverified, removing: {merged_dir}", flush=True)
        shutil.rmtree(merged_dir)
        
    # 1. Merge
    # Use CPU for merging to be safe, or GPU if preferred. Script has --device.
    merge_cmd = (
        f"python {MERGE_SCRIPT} "
        f"--base_model {BASE_MODEL} "
        f"--pt_adapter {PT_ADAPTER} "
        f"--sft_adapter {SFT_ADAPTER} "
        f"--alpha_pt {pt_w} "
        f"--alpha_sft {sft_w} "
        f"--export_dir {merged_dir} "
        f"--device cpu" 
    )
    try:
        run_cmd(merge_cmd, env=env)
        
        # Write merge_meta.json
        meta_data = {
            "alpha_pt": pt_w,
            "alpha_sft": sft_w,
            "pt_adapter": PT_ADAPTER,
            "sft_adapter": SFT_ADAPTER,
            "fingerprint": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(os.path.join(merged_dir, "merge_meta.json"), "w") as f:
            json.dump(meta_data, f, indent=2)
            
    except Exception as e:
        print(f"[GPU {gpu_id}] Merge failed: {e}")
        return

    # 2. Verify
    print(f"[GPU {gpu_id}] Verifying PT={pt_w}, SFT={sft_w}...", flush=True)
    report_path = os.path.join(merged_dir, "verify_report.json")
    verify_cmd = (
        f"python {VERIFY_SCRIPT} "
        f"--base_model {BASE_MODEL} "
        f"--merged_model {merged_dir} "
        f"--pt_adapter {PT_ADAPTER} "
        f"--sft_adapter {SFT_ADAPTER} "
        f"--alpha_pt {pt_w} "
        f"--alpha_sft {sft_w} "
        f"--save_report {report_path} "
        f"--device cuda:0"
    )
    try:
        run_cmd(verify_cmd, env=env)
        print(f"[GPU {gpu_id}] Verification SUCCESS for PT={pt_w}, SFT={sft_w}", flush=True)

        # Update meta with verification stats
        meta_path = os.path.join(merged_dir, "merge_meta.json")
        if os.path.exists(meta_path) and os.path.exists(report_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            with open(report_path, "r") as f:
                report = json.load(f)
            meta["verification_status"] = "success"
            meta["verification_stats"] = report
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[GPU {gpu_id}] Verification FAILED for PT={pt_w}, SFT={sft_w}: {e}")

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Queue of tasks
    task_queue = queue.Queue()
    for w in WEIGHTS:
        task_queue.put(w)

    def worker(gpu_id):
        while not task_queue.empty():
            try:
                pt_w, sft_w = task_queue.get_nowait()
            except queue.Empty:
                break
            
            try:
                merge_and_verify(pt_w, sft_w, gpu_id)
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing {pt_w}/{sft_w}: {e}")
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

import argparse
import json
import os
import yaml
import subprocess
import datetime
import torch

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_gpu_info():
    gpu_info = []
    try:
        # Try nvidia-smi xml or csv? Simple approach: torch.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": f"{torch.cuda.get_device_capability(i)}",
                    "mem_total": f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
                })
        else:
            gpu_info.append("No CUDA device available")
    except Exception as e:
        gpu_info.append(f"Error getting GPU info: {str(e)}")
    
    # Also try to get nvidia-smi output for driver version etc
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name,driver_version,memory.total', '--format=csv,noheader'], encoding='utf-8')
        gpu_info.append({"nvidia_smi_raw": smi.strip().split('\n')})
    except:
        pass
        
    return gpu_info

def main():
    parser = argparse.ArgumentParser(description="Create experiment manifest")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--max_samples", type=int, required=True)
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--models_list", nargs='+', help="List of models being run", default=[])
    
    args = parser.parse_args()
    
    # Load Configs
    models_cfg = load_yaml(os.path.join(args.config_dir, "models.yaml"))
    templates_cfg = load_yaml(os.path.join(args.config_dir, "templates.yaml"))
    presets_cfg = load_yaml(os.path.join(args.config_dir, "presets.yaml"))
    
    # Build Manifest
    manifest = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "backend": args.backend,
            "max_samples": args.max_samples,
            "default_max_new_tokens": args.max_new_tokens,
            "models_run": args.models_list
        },
        "environment": {
            "gpu_info": get_gpu_info(),
            "cwd": os.getcwd(),
        },
        "configurations": {
            "models": models_cfg.get('models', {}),
            "templates": templates_cfg.get('templates', {}),
            "presets": presets_cfg.get('presets', {})
        }
    }
    
    # Filter models config to only those in the list if provided
    if args.models_list:
        all_models = manifest["configurations"]["models"]
        filtered_models = {k: v for k, v in all_models.items() if k in args.models_list}
        manifest["configurations"]["models"] = filtered_models

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "manifest.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Manifest created at {output_path}")

if __name__ == "__main__":
    main()

import torch
from safetensors.torch import load_file, save_file
import os
import json
import shutil

def project_and_merge(pt_A, pt_B, sft_A, sft_B, direction, alpha, gamma):
    # DeltaW_pt
    delta_pt = pt_B @ pt_A
    
    # DeltaW_sft
    delta_sft = sft_B @ sft_A
    
    # Flatten sft
    vec_sft = delta_sft.flatten()
    
    # Project
    # sft' = sft - gamma * (d . sft) * d
    if gamma != 0:
        projection = torch.dot(direction, vec_sft) * direction
        vec_sft_prime = vec_sft - gamma * projection
        delta_sft_prime = vec_sft_prime.view_as(delta_sft)
    else:
        delta_sft_prime = delta_sft
        
    # Merge
    # merge = pt + alpha * sft'
    delta_merge = delta_pt + alpha * delta_sft_prime
    
    return delta_merge

def decompose_to_lora(delta_weight, rank):
    # delta_weight: (out, in)
    # Want B (out, r), A (r, in)
    # U, S, Vh = svd(delta_weight)
    # B = U[:, :r] @ sqrt(S)
    # A = sqrt(S) @ Vh[:r, :]
    
    U, S, Vh = torch.linalg.svd(delta_weight.float(), full_matrices=False)
    
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    
    sqrt_S = torch.diag(torch.sqrt(S))
    
    B = U @ sqrt_S
    A = sqrt_S @ Vh
    
    return A.to(delta_weight.dtype), B.to(delta_weight.dtype)

def main():
    base_dir = "/root/workspace/experiments/subspace_aware_merge_toy"
    directions_path = os.path.join(base_dir, "directions.pt")
    pt_path = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/train_pt_2025-12-17-10-58-51"
    sft_path = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/sft_medical_2025-12-15-09-16-38"
    
    print("Loading data...")
    data = torch.load(directions_path)
    directions = data["directions"]
    random_directions = data["random_directions"]
    
    pt_weights = load_file(os.path.join(pt_path, "adapter_model.safetensors"))
    sft_weights = load_file(os.path.join(sft_path, "adapter_model.safetensors"))
    
    # Read rank from config
    with open(os.path.join(pt_path, "adapter_config.json"), 'r') as f:
        config = json.load(f)
        rank = config["r"]
    
    modules = set()
    for k in pt_weights.keys():
        if "lora_A" in k:
            modules.add(k.replace(".lora_A.weight", ""))
            
    alphas = [0.5, 1.0]
    gammas = [0, 1, 5]
    
    for alpha in alphas:
        for gamma in gammas:
            if gamma == 0:
                job_types = ["naive"]
            else:
                job_types = ["proj", "rand"]
                
            for t in job_types:
                print(f"Generating merge: alpha={alpha}, gamma={gamma}, type={t}")
                
                new_weights = {}
                
                for module in modules:
                    key_A = f"{module}.lora_A.weight"
                    key_B = f"{module}.lora_B.weight"
                    
                    pt_A = pt_weights[key_A]
                    pt_B = pt_weights[key_B]
                    sft_A = sft_weights[key_A]
                    sft_B = sft_weights[key_B]
                    
                    if t == "naive":
                        d = directions[module]
                        g = 0
                    elif t == "proj":
                        d = directions[module]
                        g = gamma
                    elif t == "rand":
                        d = random_directions[module]
                        g = gamma
                    
                    delta_merge = project_and_merge(pt_A, pt_B, sft_A, sft_B, d, alpha, g)
                    
                    new_A, new_B = decompose_to_lora(delta_merge, rank)
                    
                    new_weights[key_A] = new_A
                    new_weights[key_B] = new_B
                    
                # Save
                save_dir = os.path.join(base_dir, "checkpoints", f"merge_alpha{alpha}_gamma{gamma}_{t}")
                os.makedirs(save_dir, exist_ok=True)
                
                save_file(new_weights, os.path.join(save_dir, "adapter_model.safetensors"))
                
                # Copy config
                with open(os.path.join(save_dir, "adapter_config.json"), 'w') as f:
                    json.dump(config, f, indent=2)
                    
    print("Done merging.")

if __name__ == "__main__":
    main()

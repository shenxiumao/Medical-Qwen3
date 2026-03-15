import torch
from safetensors.torch import load_file
import os

def get_delta_weight(lora_A, lora_B):
    # lora_A: (r, in)
    # lora_B: (out, r)
    # delta = B @ A
    return lora_B @ lora_A

def normalize(vec):
    return vec / (torch.norm(vec) + 1e-8)

def main():
    pt_path = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/train_pt_2025-12-17-10-58-51"
    sft_path = "/root/workspace/train/saves/Qwen3-14B-Thinking/lora/sft_medical_2025-12-15-09-16-38"
    output_path = "/root/workspace/experiments/subspace_aware_merge_toy/directions.pt"

    print(f"Loading PT from {pt_path}")
    pt_weights = load_file(os.path.join(pt_path, "adapter_model.safetensors"))
    print(f"Loading SFT from {sft_path}")
    sft_weights = load_file(os.path.join(sft_path, "adapter_model.safetensors"))

    # Identify modules
    # Keys look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    modules = set()
    for k in pt_weights.keys():
        if "lora_A" in k:
            modules.add(k.replace(".lora_A.weight", ""))
    
    directions = {}
    random_directions = {}
    
    print(f"Processing {len(modules)} modules...")
    
    for module in modules:
        key_A = f"{module}.lora_A.weight"
        key_B = f"{module}.lora_B.weight"
        
        pt_A = pt_weights[key_A]
        pt_B = pt_weights[key_B]
        sft_A = sft_weights[key_A]
        sft_B = sft_weights[key_B]
        
        delta_pt = get_delta_weight(pt_A, pt_B)
        delta_sft = get_delta_weight(sft_A, sft_B)
        
        diff = delta_pt - delta_sft
        vec_diff = diff.flatten()
        d_l = normalize(vec_diff)
        
        # Random direction
        r_l = torch.randn_like(vec_diff)
        r_l = normalize(r_l)
        
        directions[module] = d_l
        random_directions[module] = r_l
        
    torch.save({
        "directions": directions,
        "random_directions": random_directions
    }, output_path)
    print(f"Saved directions to {output_path}")

if __name__ == "__main__":
    main()

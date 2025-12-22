import argparse
import json
import os
import shutil
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def _is_attn(key: str) -> bool:
    return any(s in key for s in ["q_proj", "k_proj", "v_proj", "o_proj"])

def _is_mlp(key: str) -> bool:
    return any(s in key for s in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"])

def _scale_adapter(src_dir: str, dst_dir: str, alpha: float, alpha_attn: float | None, alpha_mlp: float | None) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    wa = load_file(os.path.join(src_dir, "adapter_model.safetensors"))
    wb = {}
    for k, v in wa.items():
        s = alpha
        if alpha_attn is not None and _is_attn(k):
            s = alpha_attn
        elif alpha_mlp is not None and _is_mlp(k):
            s = alpha_mlp
        wb[k] = v * s
    save_file(wb, os.path.join(dst_dir, "adapter_model.safetensors"))
    with open(os.path.join(src_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(dst_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        p = os.path.join(src_dir, fname)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst_dir, fname))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--pt_adapter", required=True)
    parser.add_argument("--sft_adapter", required=True)
    parser.add_argument("--alpha_pt", type=float, default=0.3)
    parser.add_argument("--alpha_sft", type=float, default=0.7)
    parser.add_argument("--alpha_attn_pt", type=float, default=None)
    parser.add_argument("--alpha_attn_sft", type=float, default=None)
    parser.add_argument("--alpha_mlp_pt", type=float, default=None)
    parser.add_argument("--alpha_mlp_sft", type=float, default=None)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    tmp_pt = os.path.join(args.export_dir, "_tmp_adapter_pt")
    tmp_sft = os.path.join(args.export_dir, "_tmp_adapter_sft")
    _scale_adapter(args.pt_adapter, tmp_pt, args.alpha_pt, args.alpha_attn_pt, args.alpha_mlp_pt)
    _scale_adapter(args.sft_adapter, tmp_sft, args.alpha_sft, args.alpha_attn_sft, args.alpha_mlp_sft)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    model = PeftModel.from_pretrained(model, tmp_pt)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, tmp_sft)
    model = model.merge_and_unload()

    os.makedirs(args.export_dir, exist_ok=True)
    model.save_pretrained(args.export_dir)
    tok.save_pretrained(args.export_dir)

if __name__ == "__main__":
    main()

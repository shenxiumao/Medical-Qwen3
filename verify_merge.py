#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-streaming verification for Qwen3 LoRA merged full model.

Checks (on sampled columns to avoid huge matmuls):
  (W_merged - W_base)[:, C]  ≈  alpha_pt*scale_pt * (B_pt @ A_pt[:,C])
                             + alpha_sft*scale_sft * (B_sft @ A_sft[:,C])

Adapter keys example:
  base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
Base keys example:
  model.layers.0.mlp.down_proj.weight

Outputs per-layer diagnostics:
  mae, max, rel, target_mean, target_max, denom

And summary includes worst key + those target stats.
"""

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
from safetensors import safe_open


# -------------------------
# Helpers: safetensors index
# -------------------------

@dataclass
class ShardedSafetensors:
    root: str
    weight_map: Dict[str, str]

    @staticmethod
    def from_model_dir(model_dir: str) -> "ShardedSafetensors":
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                data = json.load(f)
            weight_map = data.get("weight_map", {})
            if not weight_map:
                raise ValueError(f"weight_map empty in {idx}")
            return ShardedSafetensors(model_dir, weight_map)

        # fallback: single-file safetensors
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single):
            weight_map = {}
            with safe_open(single, framework="pt", device="cpu") as f:
                for k in f.keys():
                    weight_map[k] = "model.safetensors"
            return ShardedSafetensors(model_dir, weight_map)

        raise FileNotFoundError(f"Cannot find model.safetensors.index.json or model.safetensors under {model_dir}")

    def tensor(self, key: str, device: str = "cpu") -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Key not found in model weight_map: {key}")
        shard = self.weight_map[key]
        path = os.path.join(self.root, shard)
        with safe_open(path, framework="pt", device=device) as f:
            return f.get_tensor(key)


# -------------------------
# Adapter loading
# -------------------------

def find_adapter_safetensors(adapter_dir: str) -> List[str]:
    cands = []
    # LLaMAFactory: adapter_model.safetensors; sometimes multiple safetensors
    for pat in ["adapter_model.safetensors", "*.safetensors"]:
        cands.extend(glob.glob(os.path.join(adapter_dir, pat)))
    cands = sorted(set(cands))
    if not cands:
        raise FileNotFoundError(f"No .safetensors found in adapter dir: {adapter_dir}")
    return cands


def load_adapter_tensors(adapter_dir: str, device: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict]:
    files = find_adapter_safetensors(adapter_dir)
    state: Dict[str, torch.Tensor] = {}
    for fp in files:
        with safe_open(fp, framework="pt", device=device) as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    return state, cfg


# -------------------------
# Key mapping: adapter -> base
# -------------------------

def adapter_key_to_base_weight_key(adapter_lora_key: str) -> str:
    """
    Convert:
      base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
    -> model.layers.0.mlp.down_proj.weight
    """
    k = adapter_lora_key
    if k.endswith(".lora_A.weight"):
        core = k[: -len(".lora_A.weight")]
    elif k.endswith(".lora_B.weight"):
        core = k[: -len(".lora_B.weight")]
    else:
        raise ValueError(f"Not a LoRA A/B key: {k}")

    # strip base_model prefix
    if core.startswith("base_model."):
        core = core[len("base_model.") :]

    # normalize model.model -> model
    if core.startswith("model.model."):
        core = "model." + core[len("model.model.") :]

    # ensure starts with model.
    if not core.startswith("model."):
        if ".layers." in core:
            core = "model" + core[core.index(".layers.") :]
        else:
            core = "model." + core

    return core + ".weight"


# -------------------------
# LoRA scaling inference (for reporting)
# -------------------------

def infer_lora_scale(adapter_cfg: Dict) -> Tuple[bool, float, Optional[int], Optional[float]]:
    """
    Returns (rslora, scale, r, lora_alpha)
      - LoRA:    scale = alpha / r
      - RS-LoRA: scale = alpha / sqrt(r)
    """
    r = adapter_cfg.get("r", None)
    lora_alpha = adapter_cfg.get("lora_alpha", None)
    use_rslora = bool(adapter_cfg.get("use_rslora", False) or adapter_cfg.get("rslora", False))

    # fallback nested config
    if (r is None or lora_alpha is None) and "peft_config" in adapter_cfg and isinstance(adapter_cfg["peft_config"], dict):
        pc = adapter_cfg["peft_config"]
        r = pc.get("r", r)
        lora_alpha = pc.get("lora_alpha", lora_alpha)
        use_rslora = bool(pc.get("use_rslora", use_rslora))

    # defaults if missing
    scale = 1.0
    if r is not None:
        r = int(r)
        if lora_alpha is None:
            lora_alpha = float(r)
        if use_rslora:
            scale = float(lora_alpha) / (float(r) ** 0.5)
        else:
            scale = float(lora_alpha) / float(r)
    return use_rslora, float(scale), (int(r) if r is not None else None), (float(lora_alpha) if lora_alpha is not None else None)


# -------------------------
# Verification core
# -------------------------

@torch.no_grad()
def verify(
    base_dir: str,
    merged_dir: str,
    pt_dir: str,
    sft_dir: str,
    alpha_pt: float,
    alpha_sft: float,
    device: str = "cuda:0",
    sample_cols: int = 256,
    max_keys: int = 0,
    seed: int = 0,
    tol_abs_max: float = 1e-3,
    print_first_n: int = 10,
    save_report: str = None,
):
    random.seed(seed)
    torch.manual_seed(seed)

    base = ShardedSafetensors.from_model_dir(base_dir)
    merged = ShardedSafetensors.from_model_dir(merged_dir)

    pt_state, pt_cfg = load_adapter_tensors(pt_dir, device="cpu")
    sft_state, sft_cfg = load_adapter_tensors(sft_dir, device="cpu")

    pt_rslora, pt_scale, pt_r, pt_alpha = infer_lora_scale(pt_cfg)
    sft_rslora, sft_scale, sft_r, sft_alpha = infer_lora_scale(sft_cfg)

    def collect_pairs(state: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
        pairs = {}
        for k in state.keys():
            if k.endswith(".lora_A.weight"):
                core = k[: -len(".lora_A.weight")]
                kb = core + ".lora_B.weight"
                if kb in state:
                    pairs[core] = (k, kb)
        return pairs

    pt_pairs = collect_pairs(pt_state)
    sft_pairs = collect_pairs(sft_state)

    common_cores = sorted(set(pt_pairs.keys()) & set(sft_pairs.keys()))
    if max_keys and max_keys > 0:
        common_cores = common_cores[:max_keys]

    print("=== Adapter pair summary ===", flush=True)
    print(f"PT pairs  : {len(pt_pairs)}  (rslora={pt_rslora}, scale={pt_scale})", flush=True)
    print(f"SFT pairs : {len(sft_pairs)} (rslora={sft_rslora}, scale={sft_scale})", flush=True)
    print(f"Common cores to verify: {len(common_cores)}", flush=True)
    if pt_r is not None:
        print(f"PT  (r={pt_r}, alpha={pt_alpha})", flush=True)
    if sft_r is not None:
        print(f"SFT (r={sft_r}, alpha={sft_alpha})", flush=True)
    print(flush=True)

    if not common_cores:
        print("No common LoRA cores found between PT and SFT adapters.", flush=True)
        return

    dev = torch.device(device)
    verified = 0

    # Track worst by absolute max error
    worst_key = None
    worst_mae = 0.0
    worst_max = 0.0
    worst_rel = 0.0
    worst_shape = None
    worst_cols = 0

    # (patch) also track target stats for the worst key
    worst_target_mean = 0.0
    worst_target_max = 0.0
    worst_denom = 0.0

    # missing base/merged stats
    missing_in_base = 0
    missing_in_merged = 0

    # helper: normalize A/B shapes to A:[r,in], B:[out,r]
    def normalize_AB(A: torch.Tensor, B: torch.Tensor, out_dim: int, in_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("LoRA A/B not 2D")
        # already correct?
        if A.shape[1] == in_dim and B.shape[0] == out_dim and B.shape[1] == A.shape[0]:
            return A, B
        # A is [in,r]
        if A.shape[0] == in_dim and A.shape[1] != in_dim:
            A = A.t().contiguous()
        # B is [r,out]
        if B.shape[1] == out_dim and B.shape[0] != out_dim:
            B = B.t().contiguous()
        # final check
        if A.shape[1] != in_dim or B.shape[0] != out_dim or B.shape[1] != A.shape[0]:
            raise ValueError(f"Cannot align shapes. A={tuple(A.shape)}, B={tuple(B.shape)}, W={out_dim}x{in_dim}")
        return A, B

    for i, core in enumerate(common_cores, 1):
        a_pt_key, b_pt_key = pt_pairs[core]
        a_sft_key, b_sft_key = sft_pairs[core]

        base_w_key = adapter_key_to_base_weight_key(a_pt_key)

        if base_w_key not in base.weight_map:
            missing_in_base += 1
            continue
        if base_w_key not in merged.weight_map:
            missing_in_merged += 1
            continue

        # base and merged weights on CPU
        Wb = base.tensor(base_w_key, device="cpu").to(torch.float32)
        Wm = merged.tensor(base_w_key, device="cpu").to(torch.float32)

        if Wb.ndim != 2:
            del Wb, Wm
            continue

        out_dim, in_dim = Wb.shape
        k = min(sample_cols, in_dim)
        cols = torch.tensor(random.sample(range(in_dim), k), dtype=torch.long)

        target = (Wm - Wb)[:, cols].contiguous()  # [out, k]

        # load A/B CPU
        A_pt = pt_state[a_pt_key].to(torch.float32)
        B_pt = pt_state[b_pt_key].to(torch.float32)
        A_sft = sft_state[a_sft_key].to(torch.float32)
        B_sft = sft_state[b_sft_key].to(torch.float32)

        try:
            A_pt, B_pt = normalize_AB(A_pt, B_pt, out_dim, in_dim)
            A_sft, B_sft = normalize_AB(A_sft, B_sft, out_dim, in_dim)
        except Exception as e:
            if i <= print_first_n:
                print(f"[{i}/{len(common_cores)}] SKIP {base_w_key}: shape align failed: {e}")
            del Wb, Wm, target, A_pt, B_pt, A_sft, B_sft
            continue

        # move to GPU
        cols_gpu = cols.to(dev)
        target_gpu = target.to(dev)

        A_pt_gpu = A_pt.to(dev)
        B_pt_gpu = B_pt.to(dev)
        A_sft_gpu = A_sft.to(dev)
        B_sft_gpu = B_sft.to(dev)

        # B @ A[:, cols]
        delta_pt = B_pt_gpu @ A_pt_gpu[:, cols_gpu]     # [out, k]
        delta_sft = B_sft_gpu @ A_sft_gpu[:, cols_gpu]

        pred = (alpha_pt * pt_scale) * delta_pt + (alpha_sft * sft_scale) * delta_sft

        # ----- error + (patch) target stats -----
        err = (pred - target_gpu).abs()

        target_abs = target_gpu.abs()
        target_mean = target_abs.mean().item()
        target_max = target_abs.max().item()
        denom = max(target_mean, 1e-6)

        mae = err.mean().item()
        mx = err.max().item()
        rel = mae / denom
        # ---------------------------------------

        verified += 1

        # decide OK/BAD using abs max threshold; rel is diagnostic only
        ok = (mx <= tol_abs_max)
        flag = "OK " if ok else "BAD"

        if (i <= print_first_n) or (not ok):
            print(
                f"[{i:>4}/{len(common_cores)}] {flag} {base_w_key} | "
                f"mae={mae:.4e} max={mx:.4e} rel={rel:.3e} | "
                f"target_mean={target_mean:.4e} target_max={target_max:.4e} denom={denom:.4e} | "
                f"shape={out_dim}x{in_dim} cols={k}",
                flush=True
            )

        # update worst
        if mx > worst_max:
            worst_key = base_w_key
            worst_mae = mae
            worst_max = mx
            worst_rel = rel
            worst_shape = (out_dim, in_dim)
            worst_cols = k

            # (patch) record worst target stats
            worst_target_mean = target_mean
            worst_target_max = target_max
            worst_denom = denom

        # cleanup
        del Wb, Wm, target, A_pt, B_pt, A_sft, B_sft
        del cols_gpu, target_gpu, A_pt_gpu, B_pt_gpu, A_sft_gpu, B_sft_gpu
        del delta_pt, delta_sft, pred, err, target_abs
        torch.cuda.empty_cache()

    print()
    print("=== Summary ===")
    print(f"Verified keys: {verified}")
    print(f"Missing in BASE  : {missing_in_base}")
    print(f"Missing in MERGED: {missing_in_merged}")

    if worst_key is None:
        print("No keys verified (mapping/shape mismatch).")
        return

    od, idim = worst_shape
    print(f"Worst key: {worst_key}")
    print(
        f"  mae={worst_mae:.4e}  max={worst_max:.4e}  rel={worst_rel:.3e}  "
        f"shape={od}x{idim}  sampled_cols={worst_cols}"
    )
    print(
        f"  target_mean={worst_target_mean:.4e}  target_max={worst_target_max:.4e}  denom={worst_denom:.4e}"
    )

    print("\nInterpretation tips:")
    print("- Focus on absolute max/mae first (bf16 full-merge often lands around 1e-4~1e-3).")
    print("- rel can look huge when target_mean is tiny (LoRA delta nearly zero on sampled columns).")
    print("- If abs max is ~1e-2 or worse across many layers, then scaling/merge is truly wrong.")

    if save_report:
        report = {
            "verified_keys": verified,
            "missing_in_base": missing_in_base,
            "missing_in_merged": missing_in_merged,
            "worst_key": worst_key,
            "worst_mae": worst_mae,
            "worst_max": worst_max,
            "worst_rel": worst_rel,
            "worst_shape": worst_shape,
            "worst_cols": worst_cols,
            "target_mean": worst_target_mean,
            "target_max": worst_target_max,
            "denom": worst_denom,
            "ok": (worst_max <= tol_abs_max) if worst_key is not None else False
        }
        with open(save_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {save_report}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--merged_model", required=True)
    p.add_argument("--pt_adapter", required=True)
    p.add_argument("--sft_adapter", required=True)
    p.add_argument("--alpha_pt", type=float, required=True)
    p.add_argument("--alpha_sft", type=float, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample_cols", type=int, default=256)
    p.add_argument("--max_keys", type=int, default=0, help="0 means verify all common keys")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol_abs_max", type=float, default=1e-3)
    p.add_argument("--print_first_n", type=int, default=10)
    p.add_argument("--save_report", type=str, default=None, help="Path to save verification report (JSON)")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device is cuda:*")

    verify(
        base_dir=args.base_model,
        merged_dir=args.merged_model,
        pt_dir=args.pt_adapter,
        sft_dir=args.sft_adapter,
        alpha_pt=args.alpha_pt,
        alpha_sft=args.alpha_sft,
        device=args.device,
        sample_cols=args.sample_cols,
        max_keys=args.max_keys,
        seed=args.seed,
        tol_abs_max=args.tol_abs_max,
        print_first_n=args.print_first_n,
        save_report=args.save_report,
    )


if __name__ == "__main__":
    main()

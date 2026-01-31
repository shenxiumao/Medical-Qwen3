#!/usr/bin/env python3
import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

TEXT_LIKE = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "added_tokens.json",
    "README.md",
    "merge_meta.json",
}
IGNORE_PREFIX = {".git", "__pycache__"}

def sha256_file(p: Path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def list_files(root: Path) -> Dict[str, Dict]:
    out = {}
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(root))
        if any(rel.startswith(x) for x in IGNORE_PREFIX):
            continue
        st = p.stat()
        out[rel] = {"size": st.st_size, "mtime": int(st.st_mtime)}
    return out

def load_json_if_exists(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def diff_json(a, b, path="") -> List[str]:
    diffs = []
    if type(a) != type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return diffs
    if isinstance(a, dict):
        ka, kb = set(a.keys()), set(b.keys())
        for k in sorted(ka - kb):
            diffs.append(f"{path}/{k}: only in OLD")
        for k in sorted(kb - ka):
            diffs.append(f"{path}/{k}: only in NEW")
        for k in sorted(ka & kb):
            diffs += diff_json(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        n = min(len(a), len(b))
        for i in range(n):
            diffs += diff_json(a[i], b[i], f"{path}[{i}]")
    else:
        if a != b:
            diffs.append(f"{path}: {a} != {b}")
    return diffs

def load_index(model_dir: Path):
    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        raise FileNotFoundError(f"Missing model.safetensors.index.json in {model_dir}")
    return json.loads(idx.read_text())

def iter_weight_keys(index_obj) -> List[str]:
    return sorted(index_obj["weight_map"].keys())

def open_tensor(model_dir: Path, index_obj, key: str):
    shard = index_obj["weight_map"][key]
    path = model_dir / shard
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)

def take_sample(t: torch.Tensor, sample_elems: int, sample_rows: int, sample_cols: int) -> torch.Tensor:
    # Always return a contiguous float32 CPU tensor of small size.
    if t.ndim == 2:
        r = min(t.shape[0], sample_rows)
        c = min(t.shape[1], sample_cols)
        s = t[:r, :c].contiguous()
        return s.float()
    else:
        flat = t.reshape(-1)
        n = min(flat.numel(), sample_elems)
        s = flat[:n].contiguous()
        return s.float()

def stats(old_s: torch.Tensor, new_s: torch.Tensor):
    diff = new_s - old_s
    mae = diff.abs().mean().item()
    mx = diff.abs().max().item()
    denom = old_s.abs().mean().item() + 1e-8
    rel = mae / denom

    o = old_s.flatten()
    n = new_s.flatten()
    dot = torch.dot(o, n).item()
    on = torch.linalg.norm(o).item() + 1e-12
    nn = torch.linalg.norm(n).item() + 1e-12
    cos = dot / (on * nn)

    l2 = torch.linalg.norm(diff).item()
    return {"mae": mae, "max_abs": mx, "rel_mae": rel, "cosine": cos, "l2": l2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", required=True)
    ap.add_argument("--new_dir", required=True)
    ap.add_argument("--out", default="compare_report_cpu.json")
    ap.add_argument("--max_keys", type=int, default=400)
    ap.add_argument("--keys_contains", default="")
    ap.add_argument("--topk", type=int, default=30)

    # sampling knobs
    ap.add_argument("--sample_elems", type=int, default=2048, help="for 1D/nd tensors: number of elements")
    ap.add_argument("--sample_rows", type=int, default=64, help="for 2D tensors: rows")
    ap.add_argument("--sample_cols", type=int, default=64, help="for 2D tensors: cols")

    args = ap.parse_args()
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)

    report = {"old_dir": str(old_dir), "new_dir": str(new_dir), "file_diff": {}, "json_diffs": {}, "weight_compare": {}}

    # 1) file inventory diff
    old_files = list_files(old_dir)
    new_files = list_files(new_dir)
    only_old = sorted(set(old_files) - set(new_files))
    only_new = sorted(set(new_files) - set(old_files))
    common = sorted(set(old_files) & set(new_files))

    changed_size = []
    for f in common:
        if old_files[f]["size"] != new_files[f]["size"]:
            changed_size.append((f, old_files[f]["size"], new_files[f]["size"]))

    report["file_diff"] = {"only_old": only_old, "only_new": only_new, "changed_size": changed_size}

    # 2) json/config diffs
    for name in sorted(TEXT_LIKE):
        o = old_dir / name
        n = new_dir / name
        if o.exists() or n.exists():
            jo = load_json_if_exists(o)
            jn = load_json_if_exists(n)
            if jo is not None and jn is not None:
                diffs = diff_json(jo, jn, path=name)
                if diffs:
                    report["json_diffs"][name] = diffs[:2000]
            else:
                if o.exists() and n.exists():
                    so, sn = sha256_file(o), sha256_file(n)
                    report["json_diffs"][name] = {"old_sha256": so, "new_sha256": sn, "same": so == sn}
                elif o.exists():
                    report["json_diffs"][name] = {"only": "old", "sha256": sha256_file(o)}
                elif n.exists():
                    report["json_diffs"][name] = {"only": "new", "sha256": sha256_file(n)}

    # 3) weights compare (sampled)
    old_idx = load_index(old_dir)
    new_idx = load_index(new_dir)

    old_keys = set(iter_weight_keys(old_idx))
    new_keys = set(iter_weight_keys(new_idx))

    common_w = sorted(old_keys & new_keys)
    if args.keys_contains:
        common_w = [k for k in common_w if args.keys_contains in k]
    common_w = common_w[:args.max_keys]

    report["weight_compare"]["missing_in_new"] = sorted(old_keys - new_keys)[:500]
    report["weight_compare"]["missing_in_old"] = sorted(new_keys - old_keys)[:500]
    report["weight_compare"]["compared"] = len(common_w)

    diffs = []
    for k in common_w:
        try:
            ot = open_tensor(old_dir, old_idx, k)
            nt = open_tensor(new_dir, new_idx, k)
        except Exception as e:
            diffs.append((k, {"error": str(e)}))
            continue

        if ot.shape != nt.shape:
            diffs.append((k, {"shape_mismatch": [list(ot.shape), list(nt.shape)]}))
            continue

        osamp = take_sample(ot, args.sample_elems, args.sample_rows, args.sample_cols)
        nsamp = take_sample(nt, args.sample_elems, args.sample_rows, args.sample_cols)

        st = stats(osamp, nsamp)
        st["shape"] = list(ot.shape)
        diffs.append((k, st))

    valid = [(k, s) for k, s in diffs if isinstance(s, dict) and "max_abs" in s]
    valid.sort(key=lambda x: (x[1]["max_abs"], x[1]["mae"]), reverse=True)
    topk = valid[:args.topk]

    report["weight_compare"]["topk_most_different"] = topk

    # sentinel slice fingerprints (fast identity check)
    sentinel = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    fp = {}
    for k in sentinel:
        if k in old_keys and k in new_keys:
            ot = open_tensor(old_dir, old_idx, k).reshape(-1)
            nt = open_tensor(new_dir, new_idx, k).reshape(-1)
            o_bytes = ot[:4096].to(torch.float32).cpu().numpy().tobytes()
            n_bytes = nt[:4096].to(torch.float32).cpu().numpy().tobytes()
            fp[k] = {
                "old_sha256_slice": hashlib.sha256(o_bytes).hexdigest(),
                "new_sha256_slice": hashlib.sha256(n_bytes).hexdigest(),
                "same_slice": hashlib.sha256(o_bytes).hexdigest() == hashlib.sha256(n_bytes).hexdigest(),
                "shape": list(open_tensor(old_dir, old_idx, k).shape),
            }
    report["weight_compare"]["sentinel_fingerprints"] = fp

    outp = Path(args.out)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Wrote report: {outp.resolve()}\n")

    print("=== FILE DIFF ===")
    print(f"only_old={len(only_old)} only_new={len(only_new)} changed_size={len(changed_size)}")

    print("\n=== WEIGHT DIFF (SAMPLED) ===")
    print(f"compared_keys={len(common_w)}")
    if topk:
        print("worst_key:", topk[0][0], topk[0][1])

if __name__ == "__main__":
    main()

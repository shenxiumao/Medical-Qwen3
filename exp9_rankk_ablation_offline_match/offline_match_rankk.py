import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _candidate_input_paths() -> List[str]:
    basenames = [
        "rankk_norm_DeepSeek-R1-Distill-Qwen-7B.csv",
        "rankk_norm_Llama-3.1-8B-Instruct.csv",
        "rankk_norm_Qwen2.5-7B-Instruct.csv",
        "rankk_norm_Qwen3-8B.csv",
    ]
    candidates: List[str] = []
    for b in basenames:
        candidates.append(os.path.join("/mnt/data", b))
    for b in basenames:
        candidates.append(os.path.join("/root/workspace/exp9_rankk_ablation/outputs", b))
    return candidates


def _load_inputs(paths: Optional[List[str]]) -> pd.DataFrame:
    input_paths = paths[:] if paths else _candidate_input_paths()
    seen = set()
    dfs = []
    loaded_paths = []
    for p in input_paths:
        if p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        dfs.append(df)
        loaded_paths.append(p)
    if not dfs:
        raise FileNotFoundError(
            "No input CSVs found. Provide --inputs, or place rankk_norm_*.csv in /mnt/data or exp9_rankk_ablation/outputs."
        )
    df_all = pd.concat(dfs, ignore_index=True)
    required = {"model", "gamma", "k", "accuracy_mean", "entropy_mean", "margin_mean"}
    missing = sorted(list(required - set(df_all.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print("Loaded inputs:")
    for p in loaded_paths:
        print(f"  - {p}")
    return df_all


def _prep_model_df(df_all: pd.DataFrame, model: str, k: int) -> pd.DataFrame:
    df = df_all[(df_all["model"] == model) & (df_all["k"] == k)].copy()
    if "condition" in df.columns:
        df = df[df["condition"].astype(str).str.startswith("proj_k", na=False)]
    df = df.sort_values("gamma").reset_index(drop=True)
    return df


def _interp_piecewise_linear(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    def f(xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=np.float64)
        return np.interp(xq, x, y)

    return f


def _available_optional_metrics(df_all: pd.DataFrame) -> List[str]:
    candidates = [
        "strict_fail_rate",
        "format_violation_rate",
        "refusal_rate",
        "marker_hit_rate",
    ]
    present = [c for c in candidates if c in df_all.columns]
    return present


def _build_curves(df_k: pd.DataFrame, optional_metrics: List[str]) -> Dict[str, object]:
    gammas = df_k["gamma"].to_numpy(dtype=np.float64)
    curves: Dict[str, object] = {}
    curves["accuracy_mean"] = _interp_piecewise_linear(gammas, df_k["accuracy_mean"].to_numpy())
    curves["entropy_mean"] = _interp_piecewise_linear(gammas, df_k["entropy_mean"].to_numpy())
    curves["margin_mean"] = _interp_piecewise_linear(gammas, df_k["margin_mean"].to_numpy())
    for m in optional_metrics:
        vals = df_k[m].to_numpy(dtype=np.float64)
        if np.isnan(vals).all():
            continue
        curves[m] = _interp_piecewise_linear(gammas, vals)
    curves["_gamma_min"] = float(np.min(gammas))
    curves["_gamma_max"] = float(np.max(gammas))
    curves["_gamma_grid"] = gammas
    return curves


def _match_one_direction(
    model: str,
    ref_k: int,
    match_k: int,
    match_target: str,
    curves_ref: Dict[str, object],
    curves_match: Dict[str, object],
    g_ref_list: np.ndarray,
    search_grid: np.ndarray,
    optional_metrics: List[str],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    errors: List[float] = []

    f_ref = curves_ref[f"{match_target}_mean"]
    f_match = curves_match[f"{match_target}_mean"]
    match_vals_dense = f_match(search_grid)

    for g_ref in g_ref_list:
        target_val = float(f_ref(np.array([g_ref]))[0])
        diffs = np.abs(match_vals_dense - target_val)
        idx = int(np.argmin(diffs))
        g_match = float(search_grid[idx])
        match_val = float(match_vals_dense[idx])
        err = float(abs(match_val - target_val))
        errors.append(err)

        row: Dict[str, object] = {
            "model": model,
            "match_target": match_target,
            "ref_k": int(ref_k),
            "match_k": int(match_k),
            "g_ref": float(g_ref),
            "g_match": float(g_match),
            "match_error": err,
            "acc_ref": float(curves_ref["accuracy_mean"](np.array([g_ref]))[0]),
            "acc_match": float(curves_match["accuracy_mean"](np.array([g_match]))[0]),
            "entropy_ref": float(curves_ref["entropy_mean"](np.array([g_ref]))[0]),
            "entropy_match": float(curves_match["entropy_mean"](np.array([g_match]))[0]),
            "margin_ref": float(curves_ref["margin_mean"](np.array([g_ref]))[0]),
            "margin_match": float(curves_match["margin_mean"](np.array([g_match]))[0]),
        }

        for m in optional_metrics:
            if m in curves_ref and m in curves_match:
                row[f"{m}_ref"] = float(curves_ref[m](np.array([g_ref]))[0])
                row[f"{m}_match"] = float(curves_match[m](np.array([g_match]))[0])
            else:
                row[f"{m}_ref"] = np.nan
                row[f"{m}_match"] = np.nan

        rows.append(row)

    stats = {
        "mean_error": float(np.mean(errors)) if errors else float("nan"),
        "max_error": float(np.max(errors)) if errors else float("nan"),
        "frac_gt_1e-3": float(np.mean(np.array(errors) > 1e-3)) if errors else float("nan"),
        "n": float(len(errors)),
    }
    return rows, stats


def offline_match_rankk(args: argparse.Namespace) -> None:
    df_all = _load_inputs(args.inputs)
    df_all = df_all.copy()
    df_all["gamma"] = df_all["gamma"].astype(float)
    df_all["k"] = df_all["k"].astype(int)

    optional_metrics = _available_optional_metrics(df_all)

    models = sorted(df_all["model"].dropna().unique().tolist())
    required_ks = [1, 4]

    match_target = args.match_target.strip().lower()
    if match_target not in {"entropy", "margin"}:
        raise ValueError("--match_target must be one of: entropy, margin")

    step = float(args.search_step)
    search_grid = np.arange(0.0, 10.0 + 1e-9, step, dtype=np.float64)

    out_rows: List[Dict[str, object]] = []

    for model in models:
        curves_by_k: Dict[int, Dict[str, object]] = {}
        gamma_grid_by_k: Dict[int, np.ndarray] = {}

        missing_k = []
        for k in required_ks:
            df_k = _prep_model_df(df_all, model, k)
            if df_k.empty:
                missing_k.append(k)
                continue
            curves_by_k[k] = _build_curves(df_k, optional_metrics)
            gamma_grid_by_k[k] = df_k["gamma"].to_numpy(dtype=np.float64)

        if missing_k:
            print(f"Skipping model={model} (missing k={missing_k})")
            continue

        stats_lines = []
        for ref_k, match_k in [(1, 4), (4, 1)]:
            g_ref_list = gamma_grid_by_k[ref_k]
            rows, stats = _match_one_direction(
                model=model,
                ref_k=ref_k,
                match_k=match_k,
                match_target=match_target,
                curves_ref=curves_by_k[ref_k],
                curves_match=curves_by_k[match_k],
                g_ref_list=g_ref_list,
                search_grid=search_grid,
                optional_metrics=optional_metrics,
            )
            out_rows.extend(rows)
            stats_lines.append(
                f"ref_k={ref_k}->match_k={match_k}: mean={stats['mean_error']:.6g}, max={stats['max_error']:.6g}, frac>1e-3={stats['frac_gt_1e-3']:.2f}"
            )
            if stats["frac_gt_1e-3"] >= 0.5:
                print(
                    f"Warning: model={model} match_target={match_target} ref_k={ref_k}->match_k={match_k} has large mismatch (frac>1e-3={stats['frac_gt_1e-3']:.2f})."
                )
        print(f"Match error summary for model={model} target={match_target}:")
        for line in stats_lines:
            print(f"  {line}")

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        raise RuntimeError("No matched rows were produced.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df = out_df.sort_values(["model", "ref_k", "g_ref"]).reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_target", type=str, required=True, choices=["entropy", "margin"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--search_step", type=float, default=0.01)
    parser.add_argument("--inputs", nargs="*", default=None)
    args = parser.parse_args()
    offline_match_rankk(args)


if __name__ == "__main__":
    main()

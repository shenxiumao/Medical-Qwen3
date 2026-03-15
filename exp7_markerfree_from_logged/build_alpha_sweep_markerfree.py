import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "outputs_index.json"
RAW_ROOT = ROOT / "outputs_raw"
OUT_DIR = ROOT / "outputs"
OUT_CSV = OUT_DIR / "summary_markerfree_alpha.csv"


def load_index() -> List[Dict]:
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_result_files(index: List[Dict]) -> List[Path]:
    result_files: List[Path] = []
    for entry in index:
        cols = entry.get("available_columns", [])
        path = entry.get("file_path", "")
        if not path.endswith(".csv"):
            continue
        if "exp6_results_" not in path:
            continue
        if "gamma" not in cols or "y_correct" not in cols:
            continue
        has_entropy = "answer_entropy" in cols
        has_margin = "choice_margin" in cols
        has_logits = any(
            c.startswith("logit_") for c in cols
        ) or "option_logits_logged" in cols
        if not (has_entropy or has_logits or has_margin):
            continue
        result_files.append(ROOT / path)
    return result_files


def infer_direction_from_name(path: Path) -> str:
    name = path.name.lower()
    if "_real_" in name:
        return "real"
    if "_rand_" in name:
        return "rand"
    return "unknown"


def compute_entropy_margin_from_logits(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    logit_cols = []
    for base in ["logit_A", "logit_B", "logit_C", "logit_D"]:
        if base in df.columns:
            logit_cols.append(base)
    if not logit_cols:
        alt_cols = [c for c in df.columns if c.startswith("logit_") and not c.endswith("_sp") and not c.endswith("_dec")]
        alt_cols = sorted(alt_cols)
        if len(alt_cols) >= 2:
            logit_cols = alt_cols
    if not logit_cols:
        n = len(df)
        return np.full(n, np.nan), np.full(n, np.nan)
    logits = df[logit_cols].to_numpy(dtype=float)
    logits = np.nan_to_num(logits, nan=0.0)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return entropy, margin


def load_all_rows(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["direction"] = infer_direction_from_name(path)
        if "answer_entropy" in df.columns:
            ent = df["answer_entropy"].astype(float).to_numpy()
        else:
            ent = np.zeros(len(df), dtype=float)
        if "choice_margin" in df.columns:
            margin = df["choice_margin"].astype(float).to_numpy()
        else:
            margin = np.zeros(len(df), dtype=float)
        if np.allclose(ent, 0.0) or np.allclose(margin, 0.0):
            ent_logits, margin_logits = compute_entropy_margin_from_logits(df)
            if np.allclose(ent, 0.0):
                ent = ent_logits
            if np.allclose(margin, 0.0):
                margin = margin_logits
        df["entropy"] = ent
        df["margin"] = margin
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No suitable exp6_results CSV files with decision-space signals were found.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["gamma"] = all_df["gamma"].astype(float)
    all_df["entropy"] = all_df["entropy"].astype(float)
    all_df["margin"] = all_df["margin"].astype(float)
    all_df["y_correct"] = all_df["y_correct"].astype(float)
    all_df["refusal_flag"] = all_df.get("refusal_detected", 0).fillna(0).astype(int)
    all_df["format_flag"] = all_df.get("format_violation", 0).fillna(0).astype(int)
    if "marker_hit" in all_df.columns:
        all_df["marker_hit"] = all_df["marker_hit"].fillna(0).astype(float)
    return all_df


def compute_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for model, df_m in df.groupby("model"):
        base = df_m[df_m["gamma"] == 0.0]
        if base.empty:
            base = df_m
        ent = base["entropy"].replace([np.inf, -np.inf], np.nan).dropna()
        mar = base["margin"].replace([np.inf, -np.inf], np.nan).dropna()
        if ent.empty or mar.empty:
            continue
        e0 = float(np.quantile(ent.to_numpy(), 0.75))
        m0 = float(np.quantile(mar.to_numpy(), 0.25))
        thresholds[model] = {"E0": e0, "M0": m0}
    if not thresholds:
        raise RuntimeError("Failed to compute thresholds E0/M0 for any model.")
    return thresholds


def apply_mfl(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    df["MFL_indicator"] = np.nan
    for model, params in thresholds.items():
        mask = df["model"] == model
        if not mask.any():
            continue
        e0 = params["E0"]
        m0 = params["M0"]
        ent = df.loc[mask, "entropy"].to_numpy(dtype=float)
        mar = df.loc[mask, "margin"].to_numpy(dtype=float)
        ent = np.where(np.isnan(ent), e0, ent)
        mar = np.where(np.isnan(mar), m0, mar)
        refusal = df.loc[mask, "refusal_flag"].to_numpy(dtype=int)
        fmt = df.loc[mask, "format_flag"].to_numpy(dtype=int)
        cond = (ent > e0) | (mar < m0) | (refusal == 1) | (fmt == 1)
        df.loc[mask, "MFL_indicator"] = cond.astype(float)
    return df


def summarize_alpha(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df.groupby(["model", "gamma"])
    for (model, gamma), g in grouped:
        g_valid = g.replace([np.inf, -np.inf], np.nan)
        n = len(g_valid)
        if n == 0:
            continue
        accuracy = float(g_valid["y_correct"].mean())
        mfl_rate = float(g_valid["MFL_indicator"].mean())
        entropy_mean = float(g_valid["entropy"].mean())
        margin_mean = float(g_valid["margin"].mean())
        refusal_rate = float(g_valid["refusal_flag"].mean())
        format_rate = float(g_valid["format_flag"].mean())
        row = {
            "model": model,
            "alpha": float(gamma),
            "N": int(n),
            "accuracy": accuracy,
            "MFL_rate": mfl_rate,
            "entropy_mean": entropy_mean,
            "margin_mean": margin_mean,
            "refusal_rate": refusal_rate,
            "format_violation_rate": format_rate,
        }
        if "marker_hit" in g_valid.columns:
            row["marker_hit_rate"] = float(g_valid["marker_hit"].mean())
        records.append(row)
    if not records:
        raise RuntimeError("No summary records were produced.")
    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.sort_values(["model", "alpha"]).reset_index(drop=True)
    return out_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    files = select_result_files(index)
    df_all = load_all_rows(files)
    thresholds = compute_thresholds(df_all)
    df_with_mfl = apply_mfl(df_all, thresholds)
    summary = summarize_alpha(df_with_mfl)
    summary.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()

import os
import sys
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.special import softmax


def load_exp6_config() -> Dict[str, Any]:
    import yaml

    cfg_path = "/root/workspace/experiments_markerfree/exp6_correctness_defined_u/config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def find_exp8_base_logits_file(model_name: str) -> str | None:
    base_dir = "/root/workspace/exp8_temp_baseline/outputs"
    path = os.path.join(base_dir, f"exp8_base_logits_{model_name}.csv")
    if os.path.exists(path):
        return path
    return None


def find_exp6_base_logits_file(model_name: str) -> str | None:
    candidates = [
        "/root/workspace/experiments_markerfree/outputs/exp6_4",
        "/root/workspace/experiments_markerfree/outputs/exp6_3",
        "/root/workspace/experiments_markerfree/outputs/exp6_2",
        "/root/workspace/experiments_markerfree/outputs/exp6_1",
    ]
    for d in candidates:
        path = os.path.join(d, f"exp6_results_real_{model_name}.csv")
        if os.path.exists(path):
            return path
    return None


def load_gold_labels() -> Dict[Tuple[str, str], str]:
    cfg = load_exp6_config()
    sys.path.append("/root/workspace/experiments_markerfree/exp6_correctness_defined_u")
    import run_marker_irrelevant_label as exp6_mod

    N = cfg["dataset"]["N"]
    seed = cfg["seed"]
    mmlu_path = cfg["dataset"]["mmlu_path"]
    medqa_path = cfg["dataset"]["medqa_path"]

    mmlu = exp6_mod.load_mcq_data(mmlu_path, "mmlu", N, seed)
    medqa = exp6_mod.load_mcq_data(medqa_path, "medqa", N, seed)

    gold_map: Dict[Tuple[str, str], str] = {}
    for item in mmlu + medqa:
        ds = item["type"]
        sid = str(item["id"])
        gold = str(item["gold"])
        gold_map[(ds, sid)] = gold
    return gold_map


def extract_logits_and_labels(
    base_path: str, gold_map: Dict[Tuple[str, str], str] | None
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(base_path)
    if "gamma" in df.columns:
        df = df[df["gamma"] == 0].copy()
        if df.empty:
            raise ValueError(f"No gamma=0 rows found in {base_path}")

    logits_cols_sets = [
        ["logit_A_dec", "logit_B_dec", "logit_C_dec", "logit_D_dec"],
        ["logit_A_sp", "logit_B_sp", "logit_C_sp", "logit_D_sp"],
        ["logit_A", "logit_B", "logit_C", "logit_D"],
    ]

    chosen_cols: List[str] | None = None
    for cols in logits_cols_sets:
        if all(c in df.columns for c in cols):
            chosen_cols = cols
            break
    if chosen_cols is None:
        raise ValueError(
            f"No full set of option logits columns found in {base_path} "
            f"(tried {logits_cols_sets})"
        )

    rows_logits: List[List[float]] = []
    rows_gold: List[int] = []

    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    for _, row in df.iterrows():
        if "gold" in df.columns and not pd.isna(row.get("gold")):
            gold_letter = str(row.get("gold")).strip().upper()
        else:
            if gold_map is None:
                continue
            ds = str(row.get("dataset", ""))
            sid = str(row.get("id", ""))
            gold_letter = gold_map.get((ds, sid))
            if gold_letter is None:
                continue
            gold_letter = str(gold_letter).strip().upper()
        if gold_letter not in letter_to_idx:
            continue

        logits = []
        skip = False
        for c in chosen_cols:
            val = row.get(c)
            if pd.isna(val):
                skip = True
                break
            logits.append(float(val))
        if skip:
            continue

        rows_logits.append(logits)
        rows_gold.append(letter_to_idx[gold_letter])

    if not rows_logits:
        raise ValueError(f"No valid rows with logits and labels in {base_path}")

    logits_arr = np.asarray(rows_logits, dtype=np.float32)
    gold_arr = np.asarray(rows_gold, dtype=np.int64)
    return logits_arr, gold_arr


def compute_stats_for_T(
    logits: np.ndarray, gold_idx: np.ndarray, T: float
) -> Tuple[float, float, float]:
    z = logits / T
    z = z - np.max(z, axis=1, keepdims=True)
    p = softmax(z, axis=1)

    ent = -np.sum(p * np.log(p + 1e-8), axis=1)

    top2 = np.partition(p, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    pred = np.argmax(p, axis=1)
    acc = (pred == gold_idx).astype(np.float32)

    return float(ent.mean()), float(margin.mean()), float(acc.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match_target",
        type=str,
        choices=["entropy", "margin"],
        default="entropy",
    )
    parser.add_argument("--t_min", type=float, default=0.2)
    parser.add_argument("--t_max", type=float, default=4.0)
    parser.add_argument("--t_step", type=float, default=0.01)
    parser.add_argument("--t_refine_step", type=float, default=0.001)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    summary_path = os.path.join(out_dir, "summary_tempbaseline_all.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"{summary_path} not found.")

    summary_df = pd.read_csv(summary_path)
    stat_col = "entropy_mean" if args.match_target == "entropy" else "margin_mean"

    gold_map = load_gold_labels()

    t_grid = np.arange(args.t_min, args.t_max + 1e-9, args.t_step, dtype=np.float32)

    rows = []

    models = sorted(summary_df["model"].unique().tolist())

    for model_name in models:
        base_path = find_exp8_base_logits_file(model_name)
        source = "exp8"
        local_gold_map = None
        if base_path is None:
            base_path = find_exp6_base_logits_file(model_name)
            source = "exp6"
            local_gold_map = gold_map

        if base_path is None:
            print(f"[WARN] No base logits CSV found for model={model_name}, skipping.")
            continue

        print(f"[INFO] Using {source} base logits for model={model_name}: {base_path}")

        try:
            logits, gold_idx = extract_logits_and_labels(base_path, local_gold_map)
        except Exception as e:
            print(f"[WARN] Failed to load logits for model={model_name}: {e}")
            continue

        cfg = load_exp6_config()
        T0 = float(cfg["decoding"]["temperature"])

        mdf = summary_df[summary_df["model"] == model_name].copy()

        gammas = sorted(
            mdf[mdf["condition"] == "proj"]["gamma"].unique().tolist()
        )
        if not gammas:
            print(f"[WARN] No proj rows for model={model_name}, skipping.")
            continue

        for gamma in gammas:
            proj_row = mdf[
                (mdf["gamma"] == gamma) & (mdf["condition"] == "proj")
            ]
            if proj_row.empty:
                continue
            proj_row = proj_row.iloc[0]
            proj_stat = float(proj_row[stat_col])

            ent_means = []
            margin_means = []
            acc_means = []

            for T in t_grid:
                ent, marg, acc = compute_stats_for_T(logits, gold_idx, float(T))
                ent_means.append(ent)
                margin_means.append(marg)
                acc_means.append(acc)

            ent_means = np.asarray(ent_means, dtype=np.float32)
            margin_means = np.asarray(margin_means, dtype=np.float32)
            acc_means = np.asarray(acc_means, dtype=np.float32)

            if args.match_target == "entropy":
                stat_array = ent_means
            else:
                stat_array = margin_means

            idx_best = int(np.argmin(np.abs(stat_array - proj_stat)))
            T_coarse = float(t_grid[idx_best])

            refine_min = max(args.t_min, T_coarse - args.t_step)
            refine_max = min(args.t_max, T_coarse + args.t_step)
            refine_grid = np.arange(
                refine_min, refine_max + 1e-9, args.t_refine_step, dtype=np.float32
            )

            best_T = None
            best_ent = None
            best_marg = None
            best_acc = None
            best_err = None

            for T in refine_grid:
                ent_r, marg_r, acc_r = compute_stats_for_T(
                    logits, gold_idx, float(T)
                )
                stat_r = ent_r if args.match_target == "entropy" else marg_r
                err_r = abs(stat_r - proj_stat)
                if best_err is None or err_r < best_err:
                    best_err = err_r
                    best_T = float(T)
                    best_ent = float(ent_r)
                    best_marg = float(marg_r)
                    best_acc = float(acc_r)

            T_star = best_T
            ent_star = best_ent
            marg_star = best_marg
            acc_star = best_acc
            match_err = best_err

            print(
                f"[{model_name}] gamma={gamma}: "
                f"proj_{args.match_target}={proj_stat:.6f}, "
                f"tempmatch_exact_{args.match_target}={stat_array[idx_best]:.6f}, "
                f"|Δ|={match_err:.6e}"
            )
            if match_err > 1e-3:
                print(
                    f"[WARN] Match error > 1e-3 for model={model_name}, gamma={gamma}"
                )

            base_row = mdf[
                (mdf["gamma"] == gamma) & (mdf["condition"] == "base")
            ]
            if base_row.empty:
                base_row = proj_row
            else:
                base_row = base_row.iloc[0]

            rows.append(
                {
                    "model": model_name,
                    "seed": int(base_row.get("seed", 0)),
                    "N": int(base_row.get("N", 0)),
                    "gamma": gamma,
                    "condition": "base",
                    "temperature": T0,
                    "accuracy_mean": float(base_row["accuracy_mean"]),
                    "entropy_mean": float(base_row["entropy_mean"]),
                    "margin_mean": float(base_row["margin_mean"]),
                    "strict_fail_rate": float(
                        base_row.get("strict_fail_rate", np.nan)
                    )
                    if not pd.isna(base_row.get("strict_fail_rate", np.nan))
                    else np.nan,
                    "marker_hit_rate": float(
                        base_row.get("marker_hit_rate", np.nan)
                    )
                    if not pd.isna(base_row.get("marker_hit_rate", np.nan))
                    else np.nan,
                    "match_target": args.match_target,
                }
            )

            rows.append(
                {
                    "model": model_name,
                    "seed": int(proj_row.get("seed", 0)),
                    "N": int(proj_row.get("N", 0)),
                    "gamma": gamma,
                    "condition": "proj",
                    "temperature": T0,
                    "accuracy_mean": float(proj_row["accuracy_mean"]),
                    "entropy_mean": float(proj_row["entropy_mean"]),
                    "margin_mean": float(proj_row["margin_mean"]),
                    "strict_fail_rate": float(
                        proj_row.get("strict_fail_rate", np.nan)
                    )
                    if not pd.isna(proj_row.get("strict_fail_rate", np.nan))
                    else np.nan,
                    "marker_hit_rate": float(
                        proj_row.get("marker_hit_rate", np.nan)
                    )
                    if not pd.isna(proj_row.get("marker_hit_rate", np.nan))
                    else np.nan,
                    "match_target": args.match_target,
                }
            )

            rows.append(
                {
                    "model": model_name,
                    "seed": int(base_row.get("seed", 0)),
                    "N": int(base_row.get("N", 0)),
                    "gamma": gamma,
                    "condition": "tempmatch_exact",
                    "temperature": T_star,
                    "accuracy_mean": acc_star,
                    "entropy_mean": ent_star,
                    "margin_mean": marg_star,
                    "strict_fail_rate": np.nan,
                    "marker_hit_rate": np.nan,
                    "match_target": args.match_target,
                }
            )

    if not rows:
        print("[ERROR] No rows produced; check that base logits and summaries exist.")
        return

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "summary_tempbaseline_offline_exact.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved offline exact temperature baseline to {out_path}")


if __name__ == "__main__":
    main()

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_acc_delta(
    df: pd.DataFrame,
    out_png: str,
    out_table_csv: str,
    title: str,
) -> None:
    need_cols = {"model", "ref_k", "match_k", "g_ref", "acc_ref", "acc_match"}
    missing = sorted(list(need_cols - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns for acc delta plot: {missing}")

    sub = df[(df["ref_k"] == 1) & (df["match_k"] == 4)].copy()
    if sub.empty:
        raise ValueError("No rows for ref_k=1 and match_k=4.")

    models = sorted(sub["model"].dropna().unique().tolist())

    plt.figure(figsize=(8, 5))
    table_rows: List[Dict[str, object]] = []
    for model in models:
        mdf = sub[sub["model"] == model].sort_values("g_ref")
        x = mdf["g_ref"].to_numpy(dtype=float)
        y = (mdf["acc_match"] - mdf["acc_ref"]).to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=model)
        table_rows.append(
            {
                "model": model,
                "mean_acc_delta": float(np.mean(y)) if len(y) else float("nan"),
                "min_acc_delta": float(np.min(y)) if len(y) else float("nan"),
                "max_acc_delta": float(np.max(y)) if len(y) else float("nan"),
            }
        )

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("g_ref (k=1)")
    plt.ylabel("acc_delta (k4_match - k1_ref)")
    plt.title(title)
    plt.tight_layout()
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    out_table = pd.DataFrame(table_rows).sort_values("model").reset_index(drop=True)
    out_table.to_csv(out_table_csv, index=False)
    print(f"Saved {out_png}")
    print(f"Saved {out_table_csv}")


def _plot_optional_delta(
    df: pd.DataFrame,
    metric: str,
    out_png: str,
    title: str,
) -> None:
    ref_col = f"{metric}_ref"
    match_col = f"{metric}_match"
    if ref_col not in df.columns or match_col not in df.columns:
        return
    sub = df[(df["ref_k"] == 1) & (df["match_k"] == 4)].copy()
    if sub.empty:
        return
    if sub[ref_col].isna().all() or sub[match_col].isna().all():
        return

    models = sorted(sub["model"].dropna().unique().tolist())

    plt.figure(figsize=(8, 5))
    for model in models:
        mdf = sub[sub["model"] == model].sort_values("g_ref")
        x = mdf["g_ref"].to_numpy(dtype=float)
        y = (mdf[match_col] - mdf[ref_col]).to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=model)

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("g_ref (k=1)")
    plt.ylabel(f"{metric}_delta (k4_match - k1_ref)")
    plt.title(title)
    plt.tight_layout()
    plt.legend(fontsize=8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved {out_png}")


def summarize_and_plot(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    entropy_df = pd.read_csv(args.entropy_csv)
    margin_df = pd.read_csv(args.margin_csv)

    _plot_acc_delta(
        entropy_df,
        out_png=os.path.join(args.out_dir, "fig_rankk_entropy_matched_acc_delta.png"),
        out_table_csv=os.path.join(args.out_dir, "table_rankk_entropy_matched_acc_delta.csv"),
        title="Entropy-matched: Accuracy Delta (k=4 - k=1)",
    )
    _plot_acc_delta(
        margin_df,
        out_png=os.path.join(args.out_dir, "fig_rankk_margin_matched_acc_delta.png"),
        out_table_csv=os.path.join(args.out_dir, "table_rankk_margin_matched_acc_delta.csv"),
        title="Margin-matched: Accuracy Delta (k=4 - k=1)",
    )

    for metric in ["strict_fail_rate", "format_violation_rate", "refusal_rate", "marker_hit_rate"]:
        _plot_optional_delta(
            entropy_df,
            metric=metric,
            out_png=os.path.join(args.out_dir, f"fig_rankk_entropy_matched_{metric}_delta.png"),
            title=f"Entropy-matched: {metric} Delta (k=4 - k=1)",
        )
        _plot_optional_delta(
            margin_df,
            metric=metric,
            out_png=os.path.join(args.out_dir, f"fig_rankk_margin_matched_{metric}_delta.png"),
            title=f"Margin-matched: {metric} Delta (k=4 - k=1)",
        )

    ent_table = pd.read_csv(os.path.join(args.out_dir, "table_rankk_entropy_matched_acc_delta.csv"))
    mar_table = pd.read_csv(os.path.join(args.out_dir, "table_rankk_margin_matched_acc_delta.csv"))

    print("Mean accuracy delta by model (entropy-matched):")
    for _, r in ent_table.iterrows():
        print(f"  {r['model']}: mean_acc_delta={r['mean_acc_delta']:.6g}")
    print("Mean accuracy delta by model (margin-matched):")
    for _, r in mar_table.iterrows():
        print(f"  {r['model']}: mean_acc_delta={r['mean_acc_delta']:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_csv", type=str, default="outputs/rankk_offline_match_entropy.csv")
    parser.add_argument("--margin_csv", type=str, default="outputs/rankk_offline_match_margin.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    summarize_and_plot(args)


if __name__ == "__main__":
    main()

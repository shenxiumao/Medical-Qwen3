import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAS_SEABORN = True
except ModuleNotFoundError:
    sns = None
    HAS_SEABORN = False


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
SUMMARY_CSV = OUT_DIR / "summary_markerfree_alpha.csv"


def ensure_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_mfl_vs_marker_per_model(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    models = sorted(df["model"].unique())
    any_marker_line = False
    for model in models:
        sub = df[df["model"] == model].sort_values("alpha")
        xs = sub["alpha"].values
        plt.plot(xs, sub["MFL_rate"].values, marker="o", label=f"{model} - MFL")
        if "marker_hit_rate" in sub.columns:
            marker_vals = sub["marker_hit_rate"].fillna(0).values
            if np.any(marker_vals != 0):
                plt.plot(xs, marker_vals, linestyle="--", marker="s", label=f"{model} - marker")
                any_marker_line = True
    if not any_marker_line:
        print("Note: marker hits are zero under strict suppression; marker_hit_rate lines are omitted.")
    ax = plt.gca()
    ax.text(
        0.05,
        0.95,
        "α=25/40 only available for DeepSeek (extended sweep)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    plt.xlabel("alpha")
    plt.ylabel("rate")
    plt.title("Alpha sweep: marker-free MFL vs marker hits (per model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_mfl_vs_marker.png")
    plt.close()


def _common_alphas(df: pd.DataFrame):
    models = df["model"].unique()
    if len(models) == 0:
        return []
    alpha_sets = []
    for model in models:
        alphas = set(df[df["model"] == model]["alpha"].unique().tolist())
        alpha_sets.append(alphas)
    intersection = alpha_sets[0]
    for s in alpha_sets[1:]:
        intersection &= s
    return sorted(intersection)


def plot_mfl_common_grid(df: pd.DataFrame):
    common_alphas = _common_alphas(df)
    if not common_alphas:
        print("No common alphas across models; common-grid MFL plot skipped.")
        return
    plt.figure(figsize=(8, 5))
    for model in sorted(df["model"].unique()):
        sub = df[(df["model"] == model) & (df["alpha"].isin(common_alphas))].sort_values("alpha")
        if sub.empty:
            continue
        xs = sub["alpha"].values
        ys = sub["MFL_rate"].values
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("alpha")
    plt.ylabel("MFL_rate")
    plt.title("Alpha sweep (common grid): marker-free MFL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_mfl_common_grid.png")
    plt.close()


def plot_mfl_intersection_mean(df: pd.DataFrame):
    models = df["model"].unique()
    if len(models) == 0:
        print("No models found in summary; intersection plot skipped.")
        return
    alpha_sets = {}
    for model in models:
        alphas = sorted(df[df["model"] == model]["alpha"].unique().tolist())
        alpha_sets[model] = set(alphas)
    intersection = None
    for s in alpha_sets.values():
        if intersection is None:
            intersection = set(s)
        else:
            intersection &= s
    if not intersection:
        print("No common alphas across models; intersection plot skipped.")
        return
    intersection_list = sorted(intersection)
    rows = []
    for alpha in intersection_list:
        sub = df[df["alpha"] == alpha]
        mfl_vals = sub["MFL_rate"].dropna().values
        if len(mfl_vals) == 0:
            continue
        mean_mfl = float(np.mean(mfl_vals))
        std_mfl = float(np.std(mfl_vals))
        num_models = int(sub["model"].nunique())
        rows.append(
            {
                "alpha": float(alpha),
                "mean_MFL": mean_mfl,
                "std_MFL": std_mfl,
                "num_models": num_models,
            }
        )
    if not rows:
        print("No rows for intersection alphas; intersection plot skipped.")
        return
    inter_df = pd.DataFrame.from_records(rows)
    inter_df = inter_df.sort_values("alpha").reset_index(drop=True)
    inter_df.to_csv(OUT_DIR / "summary_markerfree_alpha_intersection_mean.csv", index=False)
    plt.figure(figsize=(8, 5))
    xs = inter_df["alpha"].values
    ys = inter_df["mean_MFL"].values
    yerr = inter_df["std_MFL"].values
    plt.errorbar(xs, ys, yerr=yerr, marker="o", label="mean MFL over models")
    plt.xlabel("alpha")
    plt.ylabel("MFL_rate (mean across models)")
    plt.title("Alpha sweep: intersection-only mean MFL across models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_mfl_intersection_mean.png")
    plt.close()


def plot_entropy(df: pd.DataFrame):
    by_alpha = df.groupby("alpha").agg(entropy_mean=("entropy_mean", "mean"))
    plt.figure(figsize=(8, 5))
    xs = by_alpha.index.values
    plt.plot(xs, by_alpha["entropy_mean"].values, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("mean entropy")
    plt.title("Alpha sweep: decision entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_entropy.png")
    plt.close()


def plot_margin(df: pd.DataFrame):
    by_alpha = df.groupby("alpha").agg(margin_mean=("margin_mean", "mean"))
    plt.figure(figsize=(8, 5))
    xs = by_alpha.index.values
    plt.plot(xs, by_alpha["margin_mean"].values, marker="o")
    plt.xlabel("alpha")
    plt.ylabel("mean choice margin")
    plt.title("Alpha sweep: choice margin")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_margin.png")
    plt.close()


def plot_entropy_common_grid(df: pd.DataFrame):
    if "entropy_mean" not in df.columns:
        print("entropy_mean column missing; common-grid entropy plot skipped.")
        return
    common_alphas = _common_alphas(df)
    if not common_alphas:
        print("No common alphas across models; common-grid entropy plot skipped.")
        return
    plt.figure(figsize=(8, 5))
    for model in sorted(df["model"].unique()):
        sub = df[(df["model"] == model) & (df["alpha"].isin(common_alphas))].sort_values("alpha")
        if sub.empty:
            continue
        xs = sub["alpha"].values
        ys = sub["entropy_mean"].values
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("alpha")
    plt.ylabel("entropy_mean")
    plt.title("Alpha sweep (common grid): decision entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_entropy_common_grid.png")
    plt.close()


def plot_margin_common_grid(df: pd.DataFrame):
    if "margin_mean" not in df.columns:
        print("margin_mean column missing; common-grid margin plot skipped.")
        return
    common_alphas = _common_alphas(df)
    if not common_alphas:
        print("No common alphas across models; common-grid margin plot skipped.")
        return
    plt.figure(figsize=(8, 5))
    for model in sorted(df["model"].unique()):
        sub = df[(df["model"] == model) & (df["alpha"].isin(common_alphas))].sort_values("alpha")
        if sub.empty:
            continue
        xs = sub["alpha"].values
        ys = sub["margin_mean"].values
        plt.plot(xs, ys, marker="o", label=model)
    plt.xlabel("alpha")
    plt.ylabel("margin_mean")
    plt.title("Alpha sweep (common grid): choice margin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_alpha_margin_common_grid.png")
    plt.close()


def run_sanity_checks(df: pd.DataFrame):
    if "MFL_rate" in df.columns and "accuracy" in df.columns:
        corr = df["MFL_rate"].corr(df["accuracy"])
        print(f"Correlation between MFL_rate and accuracy: {corr:.4f}")
        if corr is not None and abs(corr) > 0.95:
            print("WARNING: |corr(MFL_rate, accuracy)| > 0.95; MFL may be too aligned with accuracy.")
        if np.allclose(df["MFL_rate"].values, df["accuracy"].values):
            raise AssertionError("MFL_rate is numerically too close to accuracy; check leakage proxy definition.")
    for name in ["entropy_mean", "margin_mean"]:
        if name not in df.columns:
            continue
        print(f"Distribution of {name} by alpha:")
        for alpha, sub in df.groupby("alpha"):
            vals = sub[name].dropna().values
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            q25 = float(np.quantile(vals, 0.25))
            q75 = float(np.quantile(vals, 0.75))
            print(f"  alpha={alpha}: mean={mean:.4f}, std={std:.4f}, q25={q25:.4f}, q75={q75:.4f}")
    if "model" in df.columns and "alpha" in df.columns:
        print("Alphas per model:")
        alpha_sets = {}
        for model, sub in df.groupby("model"):
            alphas = sorted(sub["alpha"].unique().tolist())
            alpha_sets[model] = set(alphas)
            print(f"  {model}: {alphas}")
        intersection = None
        for s in alpha_sets.values():
            if intersection is None:
                intersection = set(s)
            else:
                intersection &= s
        if intersection is not None and intersection:
            print(f"Common alphas across all models: {sorted(intersection)}")
            if any(s != intersection for s in alpha_sets.values()):
                print("WARNING: Per-model plots use non-intersecting alpha grids; intersection-only mean plot is provided separately.")
        else:
            print("No common alphas across models.")


def main():
    ensure_output_dir()
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Expected summary CSV not found: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    run_sanity_checks(df)
    plot_mfl_vs_marker_per_model(df)
    plot_mfl_common_grid(df)
    plot_mfl_intersection_mean(df)
    plot_entropy(df)
    plot_margin(df)
    plot_entropy_common_grid(df)
    plot_margin_common_grid(df)


if __name__ == "__main__":
    main()

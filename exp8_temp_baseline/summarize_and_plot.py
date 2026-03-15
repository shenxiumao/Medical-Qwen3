import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def load_all_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    pattern = os.path.join(out_dir, "tempbaseline_*.csv")
    files = sorted(glob.glob(pattern))

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if not dfs:
        print("No tempbaseline_*.csv files found.")
        return None, out_dir

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df, out_dir


def plot_metric(all_df, out_dir, metric_col, fig_name):
    grouped = (
        all_df.groupby(["gamma", "condition"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["condition", "gamma"])
    )

    plt.figure(figsize=(6, 4))
    for cond, style in zip(["base", "proj", "tempmatch"], ["k--", "b-", "r-"]):
        sub = grouped[grouped["condition"] == cond]
        if sub.empty:
            continue
        plt.plot(
            sub["gamma"],
            sub[metric_col],
            style,
            marker="o",
            label=cond,
        )

    plt.xlabel("gamma")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs gamma: proj vs temp-match (base as reference)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, fig_name)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def load_offline_exact(out_dir):
    # Try with_fail first
    path_fail = os.path.join(out_dir, "summary_tempbaseline_offline_exact_with_fail.csv")
    if os.path.exists(path_fail):
        print(f"Loading exact results from {path_fail}")
        return pd.read_csv(path_fail)

    path = os.path.join(out_dir, "summary_tempbaseline_offline_exact.csv")
    if not os.path.exists(path):
        print("No summary_tempbaseline_offline_exact.csv found, skipping exact plots.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def plot_metric_exact(exact_df, out_dir, metric_col, fig_name):
    grouped = (
        exact_df.groupby(["gamma", "condition"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["condition", "gamma"])
    )

    plt.figure(figsize=(6, 4))
    for cond, style in zip(
        ["base", "proj", "tempmatch_exact"], ["k--", "b-", "r-"]
    ):
        sub = grouped[grouped["condition"] == cond]
        if sub.empty:
            continue
        plt.plot(
            sub["gamma"],
            sub[metric_col],
            style,
            marker="o",
            label=cond,
        )

    plt.xlabel("gamma")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs gamma: proj vs temp-match (offline exact)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, fig_name)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def main():
    all_df, out_dir = load_all_results()
    if all_df is None:
        return

    summary_path = os.path.join(out_dir, "summary_tempbaseline_all.csv")
    all_df.to_csv(summary_path, index=False)
    print(f"Saved aggregated CSV to {summary_path}")

    plot_metric(all_df, out_dir, "accuracy_mean", "fig_proj_vs_tempmatch_accuracy.png")

    if "strict_fail_rate" in all_df.columns:
        plot_metric(
            all_df,
            out_dir,
            "strict_fail_rate",
            "fig_proj_vs_tempmatch_strictfail.png",
        )

    plot_metric(all_df, out_dir, "entropy_mean", "fig_proj_vs_tempmatch_entropy.png")
    plot_metric(all_df, out_dir, "margin_mean", "fig_proj_vs_tempmatch_margin.png")

    exact_df = load_offline_exact(out_dir)
    if exact_df is not None:
        plot_metric_exact(
            exact_df,
            out_dir,
            "accuracy_mean",
            "fig_proj_vs_tempmatch_exact_accuracy.png",
        )
        plot_metric_exact(
            exact_df,
            out_dir,
            "entropy_mean",
            "fig_proj_vs_tempmatch_exact_entropy.png",
        )
        plot_metric_exact(
            exact_df,
            out_dir,
            "margin_mean",
            "fig_proj_vs_tempmatch_exact_margin.png",
        )

        if "strict_fail_rate" in exact_df.columns:
            plot_metric_exact(
                exact_df,
                out_dir,
                "strict_fail_rate",
                "fig_proj_vs_tempmatch_exact_fail.png",
            )

        if "format_violation_rate" in exact_df.columns:
            plot_metric_exact(
                exact_df,
                out_dir,
                "format_violation_rate",
                "fig_proj_vs_tempmatch_exact_format.png",
            )

        if "strict_fail_rate" in exact_df.columns:
            has_tmatch_strict = exact_df[
                (exact_df["condition"] == "tempmatch_exact")
                & exact_df["strict_fail_rate"].notna()
            ]
            if not has_tmatch_strict.empty:
                plot_metric_exact(
                    exact_df,
                    out_dir,
                    "strict_fail_rate",
                    "fig_proj_vs_tempmatch_exact_strictfail.png",
                )
            else:
                print(
                    "No per-example strict_fail for tempmatch_exact; "
                    "skipping exact strictfail plot."
                )


if __name__ == "__main__":
    main()


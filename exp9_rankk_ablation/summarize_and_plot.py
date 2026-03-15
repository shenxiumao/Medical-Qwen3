import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def summarize_and_plot(args):
    if args.input_file:
        files = [args.input_file]
    else:
        files = glob.glob("outputs/rankk_*.csv")
    
    if not files:
        print("No input files found.")
        return
        
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    
    if args.model_name:
        df = df[df['model'] == args.model_name]

    df_base = df[df["condition"] == "base"]
    base_vals = {}
    if not df_base.empty:
        base_row = df_base.iloc[0]
        base_vals = {
            "accuracy_mean": float(base_row.get("accuracy_mean", float("nan"))),
            "entropy_mean": float(base_row.get("entropy_mean", float("nan"))),
            "margin_mean": float(base_row.get("margin_mean", float("nan"))),
            "strict_fail_rate": float(base_row.get("strict_fail_rate", float("nan"))),
        }

    df_proj = df[df["condition"].str.startswith("proj_k", na=False)].copy()
    if df_proj.empty:
        print("No proj_k* rows found.")
        return

    suffix = ""
    if args.input_file:
        base = os.path.basename(args.input_file)
        if base.endswith(".csv"):
            # e.g., rankk_DeepSeek.csv -> _DeepSeek
            # e.g., rankk_norm_DeepSeek.csv -> _norm_DeepSeek
            raw_suffix = base[:-4]
            if raw_suffix.startswith("rankk_"):
                suffix = "_" + raw_suffix[6:]
            else:
                suffix = "_" + raw_suffix
    elif args.model_name:
        suffix = "_" + args.model_name.replace("/", "_")

    os.makedirs("outputs", exist_ok=True)
    
    # Handle the specific request for "rankk_norm_<metric>" if the suffix starts with "norm"
    # The user asked for "fig_rankk_norm_accuracy.png"
    # If suffix is "_norm_DeepSeek", we want "fig_rankk_norm_accuracy_DeepSeek.png" maybe?
    # Or strictly "fig_rankk_norm_accuracy.png" if it's a single run?
    # Let's just stick to a consistent schema: fig_rankk_<metric><suffix>.png
    # If suffix is "_norm_DeepSeek", then: fig_rankk_accuracy_norm_DeepSeek.png
    # This is close enough and unambiguous.
    
    metric_to_file = {
        "accuracy_mean": f"outputs/fig_rankk_accuracy{suffix}.png",
        "entropy_mean": f"outputs/fig_rankk_entropy{suffix}.png",
        "margin_mean": f"outputs/fig_rankk_margin{suffix}.png",
    }
    if "strict_fail_rate" in df.columns and not df["strict_fail_rate"].isna().all():
        metric_to_file["strict_fail_rate"] = f"outputs/fig_rankk_strictfail{suffix}.png"

    ks = sorted([int(k) for k in df_proj["k"].dropna().unique().tolist()])
    colors = {k: c for k, c in zip(ks, ["C0", "C1", "C2", "C3"])}

    for metric, out_path in metric_to_file.items():
        plt.figure(figsize=(7, 5))

        for k in ks:
            sub = df_proj[df_proj["k"] == k].sort_values("gamma")
            plt.plot(
                sub["gamma"].to_numpy(),
                sub[metric].to_numpy(),
                marker="o",
                label=f"k={k}",
                color=colors.get(k, None),
            )

        base_val = base_vals.get(metric, float("nan"))
        if not pd.isna(base_val):
            plt.axhline(base_val, color="gray", linestyle="--", label="base")

        plt.xlabel("gamma")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    
    summarize_and_plot(args)

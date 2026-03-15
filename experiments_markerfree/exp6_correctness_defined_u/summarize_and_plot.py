import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "exp6")
    
    # Load Results
    files = glob.glob(os.path.join(output_dir, "exp6_results_*.csv"))
    if not files:
        print("No results found.")
        return
        
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Parse filename to get direction type
        # format: exp6_results_{direction}_{model}.csv
        basename = os.path.basename(f)
        parts = basename.replace("exp6_results_", "").replace(".csv", "").split("_")
        # direction is first part (real/rand)
        direction = parts[0]
        # model is rest
        model = "_".join(parts[1:])
        
        df['direction'] = direction
        # df['model'] is already in CSV
        dfs.append(df)
        
    df_all = pd.concat(dfs)
    
    # Compute Aggregates
    agg_dict = {
        'y_correct': 'mean',
        'marker_hit': 'mean'
    }
    
    # Optional metrics
    optional_metrics = ['refusal_detected', 'format_violation', 'choice_margin', 'answer_entropy']
    available_optional_metrics = [m for m in optional_metrics if m in df_all.columns]
    
    for m in available_optional_metrics:
        agg_dict[m] = 'mean'

    # Groupby and aggregate
    grouped = df_all.groupby(['model', 'gamma', 'direction', 'dataset'])
    summary = grouped.agg(agg_dict)
    
    # Add sample size 'n'
    summary['n'] = grouped.size()
    
    summary = summary.reset_index()
    
    # Save summary
    summary.to_csv(os.path.join(output_dir, "summary_exp6_plus.csv"), index=False)
    print(f"Saved summary to {os.path.join(output_dir, 'summary_exp6_plus.csv')}")
    
    # --- Plots ---
    
    # Plot 1: Accuracy vs Gamma (Real vs Rand)
    df_plot = df_all.groupby(['model', 'gamma', 'direction']).agg({'y_correct': 'mean'}).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x="gamma", y="y_correct", hue="model", style="direction", markers=True)
    plt.title("Exp 6: Accuracy vs Gamma (Real vs Rand)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, "acc_vs_gamma_real_vs_rand.png"))
    plt.close()
    
    # Plot 2: Strict Fail vs Gamma
    df_plot_fail = df_all.groupby(['model', 'gamma', 'direction']).agg({'marker_hit': 'mean'}).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot_fail, x="gamma", y="marker_hit", hue="model", style="direction", markers=True)
    plt.title("Exp 6: Marker Hit Rate vs Gamma")
    plt.ylabel("Marker Hit Rate")
    plt.savefig(os.path.join(output_dir, "strictfail_vs_gamma_real_vs_rand.png"))
    plt.close()

    # --- Delta Plots (Real - Rand) ---
    # Pivot to align Real and Rand columns
    # We need to pivot on direction: columns 'y_correct', 'choice_margin', etc. become (metric, direction)
    
    # Filter for models that have both real and rand
    # Group by model, gamma
    metrics_to_plot = {
        'choice_margin': 'Delta Choice Margin (Real - Rand)',
        'format_violation': 'Delta Format Violation (Real - Rand)',
        'refusal_detected': 'Delta Refusal Rate (Real - Rand)'
    }
    
    # Only plot metrics that exist in data
    metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if k in df_all.columns}
    
    if not metrics_to_plot:
        print("No optional metrics found for delta plots.")
    else:
        # Pre-aggregate across datasets first
        agg_dict_delta = {m: 'mean' for m in metrics_to_plot.keys()}
        df_agg = df_all.groupby(['model', 'gamma', 'direction']).agg(agg_dict_delta).reset_index()
        
        # Pivot
        df_pivot = df_agg.pivot(index=['model', 'gamma'], columns='direction', values=list(metrics_to_plot.keys()))
        
        # Calculate Deltas
        # Structure: df_pivot.columns is MultiIndex: (metric, direction)
        # We want metric_Real - metric_Rand
        
        # Check if we have both 'real' and 'rand' for models
        if 'real' in df_pivot.columns.get_level_values(1) and 'rand' in df_pivot.columns.get_level_values(1):
            for metric, title in metrics_to_plot.items():
                try:
                    real_vals = df_pivot[(metric, 'real')]
                    rand_vals = df_pivot[(metric, 'rand')]
                    delta = real_vals - rand_vals
                    
                    # Create a DataFrame for plotting
                    df_delta = delta.reset_index()
                    df_delta.columns = ['model', 'gamma', 'delta']
                    
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(data=df_delta, x="gamma", y="delta", hue="model", markers=True)
                    plt.title(f"Exp 6: {title}")
                    plt.ylabel(f"Delta {metric}")
                    plt.axhline(0, color='gray', linestyle='--')
                    plt.savefig(os.path.join(output_dir, f"delta_{metric}_real_minus_rand.png"))
                    plt.close()
                    print(f"Generated plot for {metric}")
                except KeyError as e:
                    print(f"Skipping plot for {metric}: Missing data ({e})")
        else:
            print("Skipping delta plots: 'real' and 'rand' directions not both present.")

    
    print("Plots generated.")

if __name__ == "__main__":
    main()

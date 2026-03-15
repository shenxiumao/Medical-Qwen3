import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def make_plots(output_dir, plot_dir=None):
    summary_path = os.path.join(output_dir, "summary_by_cell.csv")
    if not os.path.exists(summary_path):
        print("Summary CSV not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    if plot_dir:
        plots_dir = plot_dir
    else:
        # Fallback to sibling 'plots' directory if not specified
        parent = os.path.dirname(output_dir.rstrip(os.sep))
        if not parent: parent = "."
        plots_dir = os.path.join(parent, "plots")
        
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to {plots_dir}")
    
    # Combine template+preset for Y-axis
    df['config_label'] = df['template_name'] + " + " + df['preset_name']
    
    models = df['model_id'].unique()
    configs = df['config_label'].unique()
    
    # --- 1. Heatmaps ---
    metrics_to_plot = ['think_tag_rate', 'strict_fail_rate', 'leakage_ratio_mean']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 8))
        
        # Pivot table
        pivot = df.pivot(index='config_label', columns='model_id', values=metric)
        
        # Plot
        im = plt.imshow(pivot.values, cmap='viridis', aspect='auto')
        plt.colorbar(im)
        
        # Ticks
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
        plt.yticks(range(len(pivot.index)), pivot.index)
        
        # Values in cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                text = plt.text(j, i, f"{val:.2f}",
                               ha="center", va="center", color="w" if val < 0.5 else "k")
                               
        plt.title(f"Heatmap of {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heatmap_{metric}.png"))
        plt.close()
        
    print("Generated heatmaps.")
    
    # --- 2. Bar Chart: Plain vs Soft vs Strict (averaged across presets) ---
    # We want to see how template affects leakage for each model
    # Aggregating over presets
    
    df_templ = df.groupby(['model_id', 'template_name'])['leakage_ratio_mean'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    
    # Bar width
    width = 0.2
    templates = ['plain', 'nothink_soft', 'nothink_strict']
    x = np.arange(len(models))
    
    for i, tmpl in enumerate(templates):
        subset = df_templ[df_templ['template_name'] == tmpl]
        # Align with models order
        vals = []
        for m in models:
            v = subset[subset['model_id'] == m]['leakage_ratio_mean'].values
            vals.append(v[0] if len(v) > 0 else 0)
            
        plt.bar(x + i*width, vals, width, label=tmpl)
        
    plt.xlabel('Model')
    plt.ylabel('Mean Leakage Ratio')
    plt.title('Leakage Ratio by Template (Avg over Presets)')
    plt.xticks(x + width, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "bar_template_comparison.png"))
    plt.close()
    
    print("Generated bar chart.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()
    
    make_plots(args.output_dir, args.plot_dir)

import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    
    # Analyze 'plain' template as baseline for reasoning representation
    file_path = os.path.join(args.activations_dir, f"model={args.model_name}__template=plain__preset=PresetB.npz")
    
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return
        
    data = np.load(file_path)
    hs = data['hidden_states'] # [N, L, H]
    num_layers = hs.shape[1]
    
    explained_variance_ratios = []
    pc1_variance = []
    
    print(f"Calculating PCA for {args.model_name}...")
    for l in range(num_layers):
        X = hs[:, l, :]
        # Standardize? Usually PCA handles centered data. 
        # We'll let sklearn handle centering.
        
        pca = PCA(n_components=10)
        pca.fit(X)
        
        # Explained variance ratio of top component
        explained_variance_ratios.append(pca.explained_variance_ratio_[0])
        pc1_variance.append(pca.explained_variance_[0])
        
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame({
        'layer': range(num_layers), 
        'pc1_explained_ratio': explained_variance_ratios,
        'pc1_variance': pc1_variance
    })
    csv_path = os.path.join(args.output_dir, f"{args.model_name}_pca.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['pc1_explained_ratio'], marker='o', label='PC1 Ratio')
    plt.title(f"Layer-wise PCA: PC1 Explained Variance Ratio ({args.model_name})")
    plt.xlabel("Layer")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_pca_ratio.png"))
    plt.close()
    
    print(f"Saved PCA results to {args.output_dir}")

if __name__ == "__main__":
    main()

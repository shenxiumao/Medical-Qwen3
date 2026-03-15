import argparse
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def center_gram(x):
    # x: [N, features]
    # K = x @ x.T
    # Centering K
    n = x.shape[0]
    gram = np.dot(x, x.T)
    H = np.eye(n) - np.ones((n, n)) / n
    return np.dot(np.dot(H, gram), H)

def cka(gram_x, gram_y):
    # Centered Kernel Alignment
    # gram_x, gram_y are centered gram matrices
    scaled_hsic = np.trace(np.dot(gram_x, gram_y))
    norm_x = np.sqrt(np.trace(np.dot(gram_x, gram_x)))
    norm_y = np.sqrt(np.trace(np.dot(gram_y, gram_y)))
    return scaled_hsic / (norm_x * norm_y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    
    # We expect files like:
    # model={name}__template=plain__preset=PresetB.npz
    # model={name}__template=nothink_strict__preset=PresetB.npz
    
    file_plain = os.path.join(args.activations_dir, f"model={args.model_name}__template=plain__preset=PresetB.npz")
    file_strict = os.path.join(args.activations_dir, f"model={args.model_name}__template=nothink_strict__preset=PresetB.npz")
    
    if not os.path.exists(file_plain) or not os.path.exists(file_strict):
        print(f"Missing files for model {args.model_name}")
        return
        
    data_plain = np.load(file_plain)
    data_strict = np.load(file_strict)
    
    hs_plain = data_plain['hidden_states'] # [N, L, H]
    hs_strict = data_strict['hidden_states'] # [N, L, H]
    
    # Ensure same number of samples
    n_samples = min(hs_plain.shape[0], hs_strict.shape[0])
    hs_plain = hs_plain[:n_samples]
    hs_strict = hs_strict[:n_samples]
    
    num_layers = hs_plain.shape[1]
    
    cka_scores = []
    
    print(f"Calculating CKA for {args.model_name} ({num_layers} layers)...")
    for l in range(num_layers):
        X = hs_plain[:, l, :]
        Y = hs_strict[:, l, :]
        
        # Center Gram matrices
        Gx = center_gram(X)
        Gy = center_gram(Y)
        
        score = cka(Gx, Gy)
        cka_scores.append(score)
        
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame({'layer': range(num_layers), 'cka': cka_scores})
    csv_path = os.path.join(args.output_dir, f"{args.model_name}_cka.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['cka'], marker='o')
    plt.title(f"Layer-wise CKA: Plain vs Strict ({args.model_name})")
    plt.xlabel("Layer")
    plt.ylabel("CKA Similarity")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_cka.png"))
    plt.close()
    
    print(f"Saved CKA results to {args.output_dir}")

if __name__ == "__main__":
    main()

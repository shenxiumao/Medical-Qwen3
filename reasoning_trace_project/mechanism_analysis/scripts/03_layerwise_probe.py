import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    
    # Use 'plain' template data which should have mixed leaky/non-leaky samples
    file_path = os.path.join(args.activations_dir, f"model={args.model_name}__template=plain__preset=PresetB.npz")
    
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        return
        
    data = np.load(file_path)
    hs = data['hidden_states'] # [N, L, H]
    labels = data['labels'] # [N] (1=leaky, 0=clean)
    
    # Check class balance
    print(f"Labels distribution: {np.bincount(labels)}")
    if len(np.unique(labels)) < 2:
        print("Error: Only one class present in labels. Cannot run probing.")
        return
        
    num_layers = hs.shape[1]
    
    aucs = []
    accs = []
    
    print(f"Running Probing for {args.model_name}...")
    for l in range(num_layers):
        X = hs[:, l, :]
        y = labels
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Linear Probe
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Eval
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5
            
        accs.append(acc)
        aucs.append(auc)
        
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame({
        'layer': range(num_layers), 
        'accuracy': accs,
        'auc': aucs
    })
    csv_path = os.path.join(args.output_dir, f"{args.model_name}_probe.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['layer'], df['auc'], marker='o', label='AUC')
    plt.plot(df['layer'], df['accuracy'], linestyle='--', alpha=0.7, label='Accuracy')
    plt.title(f"Layer-wise Probing Performance ({args.model_name})")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_probe.png"))
    plt.close()
    
    print(f"Saved Probing results to {args.output_dir}")

if __name__ == "__main__":
    main()

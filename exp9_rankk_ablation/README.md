
# Exp9 Rank-k Ablation

This experiment extends the rank-1 logit projection intervention to rank-k (k=1, 4).
It uses Residual PCA to ensure backward compatibility:
- First component is the same `u_corr` (difference of means) as in Exp6.
- Subsequent k-1 components are PC1..PC(k-1) of the data projected onto the orthogonal complement of `u_corr`.
- All k components are orthonormalized via QR decomposition.

## Automation

To run the entire pipeline for all configured models (Llama-3.1, DeepSeek, Qwen3, Qwen2.5), use the provided script:

```bash
# Standard Rank-k Ablation
./run_all.sh

# Normalized Rank-k Ablation (z' = z - (gamma/k) U U^T z)
./run_all_norm.sh
```

This script will automatically:
1. Build $U_k$ matrices for k=1 and k=4.
2. Run the rank-k ablation evaluation (with normalization if using `run_all_norm.sh`).
3. Generate summary plots in `outputs/`.
4. Log all output to stdout (or you can redirect to a file).

## Manual Usage

1. Build Orthonormal Matrix U_k:
```bash
# For k=1
/opt/miniconda3/envs/lit_reason/bin/python build_Uk.py --model_path <path> --model_name DeepSeek --k 1 --seed 0 --N_dir 500

# For k=4
/opt/miniconda3/envs/lit_reason/bin/python build_Uk.py --model_path <path> --model_name DeepSeek --k 4 --seed 0 --N_dir 500
```

2. Run Ablation:
```bash
# This will run for k=1 and k=4, sweeping gamma=[0,2,5,10]
/opt/miniconda3/envs/lit_reason/bin/python run_rankk.py \
    --model_path <path> \
    --model_name DeepSeek \
    --k_list "1,4" \
    --gamma_grid "0,2,5,10" \
    --N 500 \
    --seed 0 \
    --Uk_path "outputs/Uk_DeepSeek_k{k}.npz" \
    --out outputs/rankk_DeepSeek.csv
```

3. Summarize and Plot:
```bash
/opt/miniconda3/envs/lit_reason/bin/python summarize_and_plot.py --input_file outputs/rankk_DeepSeek.csv
```

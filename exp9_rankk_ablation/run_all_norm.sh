#!/bin/bash
set -e

# Use the specific python environment
PYTHON="/opt/miniconda3/envs/lit_reason/bin/python"

# Define models and their paths
# Note: Using associative array for mapping model names to paths
declare -A models
models["DeepSeek-R1-Distill-Qwen-7B"]="/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
models["Llama-3.1-8B-Instruct"]="/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct"
models["Qwen3-8B"]="/root/workspace/model/Qwen/Qwen3-8B"
models["Qwen2.5-7B-Instruct"]="/root/workspace/model/Qwen/Qwen2.5-7B-Instruct"

# Create output directory
mkdir -p outputs

# Iterate over models
for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"
    echo "========================================================"
    echo "Processing Model (Normalized): $model_name"
    echo "Path: $model_path"
    echo "========================================================"
    
    # Generate a safe filename suffix (replace / with _)
    safe_name="${model_name//\//_}"
    
    # 1. Build Uk for k=1 (Reuse if exists, but for safety run it)
    # Actually, if we already have Uk files, we can skip this step to save time
    # But since the previous run might have failed or been interrupted, let's check
    if [ ! -f "outputs/Uk_${safe_name}_k1.npz" ]; then
        echo "[1/4] Building Uk (k=1)..."
        $PYTHON build_Uk.py \
            --model_path "$model_path" \
            --model_name "$model_name" \
            --k 1 \
            --seed 0 \
            --N_dir 500
    else
        echo "[1/4] Uk (k=1) exists, skipping build..."
    fi
        
    # 2. Build Uk for k=4
    if [ ! -f "outputs/Uk_${safe_name}_k4.npz" ]; then
        echo "[2/4] Building Uk (k=4)..."
        $PYTHON build_Uk.py \
            --model_path "$model_path" \
            --model_name "$model_name" \
            --k 4 \
            --seed 0 \
            --N_dir 500
    else
        echo "[2/4] Uk (k=4) exists, skipping build..."
    fi
        
    # 3. Run Rank-k Ablation (Normalized)
    # Note: k_list "1,4" means we test base (k=0), k=1, and k=4
    echo "[3/4] Running Rank-k Ablation (Normalized)..."
    $PYTHON run_rankk.py \
        --model_path "$model_path" \
        --model_name "$model_name" \
        --k_list "1,4" \
        --gamma_grid "0,2,5,10" \
        --N 500 \
        --seed 0 \
        --Uk_path "outputs/Uk_${safe_name}_k{k}.npz" \
        --out "outputs/rankk_norm_${safe_name}.csv" \
        --normalize
        
    # 4. Summarize and Plot
    echo "[4/4] Plotting results..."
    $PYTHON summarize_and_plot.py \
        --input_file "outputs/rankk_norm_${safe_name}.csv"
        
    echo "Finished processing $model_name"
    echo ""
done

echo "All models processed successfully (Normalized)!"

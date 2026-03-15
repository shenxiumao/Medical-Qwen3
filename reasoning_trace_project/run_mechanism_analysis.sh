#!/bin/bash

# Configuration
MODELS=(
    "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "/root/workspace/model/Qwen/Qwen3-8B"
    "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct"
)

# Extract basenames for simpler handling
MODEL_NAMES=("DeepSeek-R1-Distill-Qwen-7B" "Qwen3-8B" "Llama-3.1-8B-Instruct")

MAX_SAMPLES=200
OUTPUT_DIR="mechanism_analysis/output"
SCRIPT_DIR="mechanism_analysis/scripts"

mkdir -p $OUTPUT_DIR

# 1. Collect Activations
echo "=================================================="
echo "Step 1: Collecting Activations"
echo "=================================================="

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo "Processing Model: $MODEL_NAME"
    
    # Run for 'plain' template
    echo "  - Running Template: plain"
    python $SCRIPT_DIR/00_collect_activations.py \
        --model_path "$MODEL_PATH" \
        --template_name plain \
        --preset_name PresetB \
        --output_dir "$OUTPUT_DIR/activations" \
        --max_samples $MAX_SAMPLES \
        --seed 42
        
    # Run for 'nothink_strict' template
    echo "  - Running Template: nothink_strict"
    python $SCRIPT_DIR/00_collect_activations.py \
        --model_path "$MODEL_PATH" \
        --template_name nothink_strict \
        --preset_name PresetB \
        --output_dir "$OUTPUT_DIR/activations" \
        --max_samples $MAX_SAMPLES \
        --seed 42
        
    echo ""
done

# 2. Run Analyses
echo "=================================================="
echo "Step 2: Running Analyses (CKA, PCA, Probing)"
echo "=================================================="

ANALYSIS_OUT="$OUTPUT_DIR/analysis"
mkdir -p $ANALYSIS_OUT

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Analyzing Model: $MODEL_NAME"
    
    # CKA
    echo "  - CKA..."
    python $SCRIPT_DIR/01_layerwise_cka.py \
        --activations_dir "$OUTPUT_DIR/activations" \
        --output_dir "$ANALYSIS_OUT/cka" \
        --model_name "$MODEL_NAME"
        
    # PCA
    echo "  - PCA..."
    python $SCRIPT_DIR/02_layerwise_pca.py \
        --activations_dir "$OUTPUT_DIR/activations" \
        --output_dir "$ANALYSIS_OUT/pca" \
        --model_name "$MODEL_NAME"
        
    # Probing
    echo "  - Probing..."
    python $SCRIPT_DIR/03_layerwise_probe.py \
        --activations_dir "$OUTPUT_DIR/activations" \
        --output_dir "$ANALYSIS_OUT/probe" \
        --model_name "$MODEL_NAME"
        
    echo ""
done

echo "Done! Results in $OUTPUT_DIR"

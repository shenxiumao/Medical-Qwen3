#!/bin/bash

# Default arguments
MAX_SAMPLES=${1:-500} # Default to 500 samples if not provided

# 8 Models: 4 original + 4 new
# Strategy:
# - gemma-4b: Data Parallelism (TP=1x4)
# - All others: Tensor Parallelism (TP=4x1)
MODELS=("gemma-270m" "qwen-7b" "llama-8b" "qwen3-8b" "ds-r1-qwen3-8b" "ds-r1-distill-qwen-7b" "ds-r1-distill-llama-8b" "gemma-4b")

# Project root directory
PROJECT_ROOT="/root/workspace/reasoning_trace_project"
cd $PROJECT_ROOT

# Separate directories for the 8-model run
OUTPUT_DIR="output_8models"
LOG_DIR="logs_8models"
PLOT_DIR="plots_8models"

echo "Starting 8-model experiment with MAX_SAMPLES=$MAX_SAMPLES"
echo "Models: ${MODELS[@]}"
echo "Output Directory: $OUTPUT_DIR"
echo "Log Directory: $LOG_DIR"
echo "Plot Directory: $PLOT_DIR"

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $PLOT_DIR

# Clean up previous output in the NEW directory
echo "Cleaning up previous output in $OUTPUT_DIR..."
rm -f $OUTPUT_DIR/*.jsonl $OUTPUT_DIR/*.json $OUTPUT_DIR/*.csv

# Create Manifest
echo "Creating Manifest..."
python scripts/create_manifest.py \
    --output_dir $OUTPUT_DIR \
    --config_dir configs \
    --max_samples $MAX_SAMPLES \
    --backend vllm \
    --max_new_tokens 512 \
    --models_list "${MODELS[@]}"

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Running Model: $MODEL"
    echo "=================================================="
    
    if [ "$MODEL" == "gemma-4b" ]; then
        echo "  -> Using Data Parallelism (4 Shards x TP=1) for $MODEL"
        pids=()
        
        # Launch 4 shards in parallel using vLLM (TP=1 per shard)
        for i in {0..3}; do
            echo "  - Starting Shard $i on GPU $i..."
            
            # We use CUDA_VISIBLE_DEVICES to isolate the GPU
            CUDA_VISIBLE_DEVICES=$i python scripts/01_generate_vllm.py \
                --models $MODEL \
                --gpu_id 0 \
                --shard_id $i \
                --num_shards 4 \
                --tensor_parallel_size 1 \
                --gpu_memory_utilization 0.90 \
                --max_samples $MAX_SAMPLES \
                --shuffle \
                --output_dir $OUTPUT_DIR \
                --config_dir configs \
                > $LOG_DIR/${MODEL}_shard${i}.log 2>&1 &
            
            pids+=($!)
        done
        
        # Wait for all shards of this model to finish
        for pid in "${pids[@]}"; do
            wait $pid
        done
        
    else
        echo "  -> Using Tensor Parallelism (1 Shard x TP=4) for $MODEL"
        
        # Launch 1 instance using all 4 GPUs with TP=4
        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/01_generate_vllm.py \
            --models $MODEL \
            --gpu_id 0 \
            --shard_id 0 \
            --num_shards 1 \
            --tensor_parallel_size 4 \
            --gpu_memory_utilization 0.90 \
            --max_samples $MAX_SAMPLES \
            --shuffle \
            --output_dir $OUTPUT_DIR \
            --config_dir configs \
            > $LOG_DIR/${MODEL}_tp4.log 2>&1
            
    fi
        
    echo "Model $MODEL finished."
    echo ""
done

echo "=================================================="
echo "All models finished. Starting post-processing..."
echo "=================================================="

# Post-processing using the new output directory

echo "Merging shards..."
# Merge script needs to know where to look and where to write
# Currently 02_merge_shards.py takes --output_dir as both input source and output dest
python scripts/02_merge_shards.py --output_dir $OUTPUT_DIR

echo "Extracting stats..."
python scripts/03_extract_stats.py --output_dir $OUTPUT_DIR --top_n 10

echo "Generating plots..."
# Plot script needs to read from OUTPUT_DIR and write to PLOT_DIR
python scripts/04_make_plots.py --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR

echo "Done! Check $OUTPUT_DIR and $PLOT_DIR directories."

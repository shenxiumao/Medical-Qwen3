#!/bin/bash

# Default arguments
MAX_SAMPLES=${1:-500} # Default to 500 samples if not provided
# Modified order: gemma-4b is last
MODELS=("gemma-270m" "qwen-7b" "llama-8b" "gemma-4b")

# Project root directory
PROJECT_ROOT="/root/workspace/reasoning_trace_project"
cd $PROJECT_ROOT

echo "Starting full experiment with MAX_SAMPLES=$MAX_SAMPLES"
echo "Strategy: TP=4 for most models, Data Parallelism (TP=1x4) for gemma-4b"
echo "Models to run: ${MODELS[@]}"
echo "Log directory: $PROJECT_ROOT/logs"
mkdir -p logs
mkdir -p output

# Clean up previous output
echo "Cleaning up previous output..."
rm -f output/*.jsonl output/*.json output/*.csv

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
                --output_dir output \
                --config_dir configs \
                > logs/${MODEL}_shard${i}.log 2>&1 &
            
            pids+=($!)
        done
        
        # Wait for all shards of this model to finish
        for pid in "${pids[@]}"; do
            wait $pid
        done
        
    else
        echo "  -> Using Tensor Parallelism (1 Shard x TP=4) for $MODEL"
        
        # Launch 1 instance using all 4 GPUs with TP=4
        # Note: No background (&) here, we run sequentially model by model
        # or we could wait, but since we use all GPUs, we must wait anyway.
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/01_generate_vllm.py \
            --models $MODEL \
            --gpu_id 0 \
            --shard_id 0 \
            --num_shards 1 \
            --tensor_parallel_size 4 \
            --gpu_memory_utilization 0.90 \
            --max_samples $MAX_SAMPLES \
            --shuffle \
            --output_dir output \
            --config_dir configs \
            > logs/${MODEL}_tp4.log 2>&1
            
    fi
        
    echo "Model $MODEL finished."
    echo ""
done

echo "=================================================="
echo "All models finished. Starting post-processing..."
echo "=================================================="

# Post-processing
# Merging is still useful to standardize the filename to predictions_merged.jsonl
echo "Merging shards..."
python scripts/02_merge_shards.py --output_dir output

echo "Extracting stats..."
python scripts/03_extract_stats.py --output_dir output --top_n 10

echo "Generating plots..."
python scripts/04_make_plots.py --output_dir output

echo "Done! Check output/ and plots/ directories."

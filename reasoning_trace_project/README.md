# Reasoning Trace Reactivation & Template Suppression Experiment

This project evaluates the persistence of "reasoning traces" (internal thought processes) in LLMs under various suppression templates and decoding strategies. It focuses on detecting whether models leak reasoning steps (e.g., `<think>` tags or "Step-by-step" markers) when explicitly instructed not to.

## Project Structure

- `scripts/`: Python scripts for generation, merging, stats, and plotting.
- `configs/`: YAML configurations for models, presets, and templates.
- `output/`: Raw predictions and merged JSONL files.
- `plots/`: Generated charts.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Quick Test (Gemma-270M)

Verify the pipeline works with the smallest model:

```bash
python scripts/01_generate.py \
    --models gemma-270m \
    --max_samples 10 \
    --gpu_id 0
```

### 2. Main Experiment (Gemma-4B)

Run the main evaluation:

```bash
python scripts/01_generate.py \
    --models gemma-4b \
    --max_samples 500 \
    --gpu_id 0
```

### 3. Parallel Execution (4 GPUs)

To run on 4 GPUs in parallel, open 4 terminal sessions (or use tmux) and run one command in each:

**Shard 0 (GPU 0):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/01_generate.py \
    --models gemma-4b qwen-7b llama-8b \
    --gpu_id 0 \
    --shard_id 0 \
    --num_shards 4 \
    --max_samples 500
```

**Shard 1 (GPU 1):**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/01_generate.py \
    --models gemma-4b qwen-7b llama-8b \
    --gpu_id 0 \
    --shard_id 1 \
    --num_shards 4 \
    --max_samples 500
```

*(Note: Inside the container, `CUDA_VISIBLE_DEVICES=1` makes physical GPU 1 appear as device 0 to the script, so keep `--gpu_id 0` in the python arg if isolating via env var, OR set `--gpu_id 1` and don't use env var if the script handles it. The script supports `--gpu_id` argument directly.)*

**Recommended (using script argument):**

```bash
# Terminal 1
python scripts/01_generate.py --models gemma-4b --gpu_id 0 --shard_id 0 --num_shards 4 --max_samples 500 &

# Terminal 2
python scripts/01_generate.py --models gemma-4b --gpu_id 1 --shard_id 1 --num_shards 4 --max_samples 500 &

# Terminal 3
python scripts/01_generate.py --models gemma-4b --gpu_id 2 --shard_id 2 --num_shards 4 --max_samples 500 &

# Terminal 4
python scripts/01_generate.py --models gemma-4b --gpu_id 3 --shard_id 3 --num_shards 4 --max_samples 500 &
```

### 4. Post-Processing

After all generation jobs finish:

**Merge Shards:**
```bash
python scripts/02_merge_shards.py --output_dir output
```

**Extract Statistics:**
```bash
python scripts/03_extract_stats.py --output_dir output --top_n 10
```

**Generate Plots:**
```bash
python scripts/04_make_plots.py --output_dir output
```

Results will be in `output/` (CSVs, JSONLs) and `plots/` (PNGs).

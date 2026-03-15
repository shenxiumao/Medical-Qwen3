Stage-2 temperature / logit-scaling baseline (Exp8)
===================================================

This folder implements a temperature-matched baseline to test whether the projection
intervention from Exp6 can be explained purely as an effective temperature change
in the multiple-choice decision distribution.

Files
-----
- `run_temp_baseline.py`: runs the baseline for a single model.
- `summarize_and_plot.py`: aggregates all temp-baseline CSVs and produces plots.
- `outputs/`: directory where CSVs and figures are written (created automatically).

Core idea
---------
For each gamma in a user-specified grid, we compare:
- **base**: projection OFF (gamma=0), default decoding temperature `T0` from
  `exp6_correctness_defined_u/config.yaml`.
- **proj**: projection ON at this gamma using the correctness-defined direction
  `u_corr` from Exp6, evaluated at the same `T0`.
- **tempmatch**: projection OFF (gamma=0) at a temperature `T*` selected from a
  candidate grid so that the marker-free decision-space statistic (entropy or
  margin over option logits) is as close as possible to the corresponding proj
  statistic at that gamma.

Metrics are computed in the same MCQ answer-only setting as Exp6, using the
option logits at the detected answer decision step.

How to run (4 models, tp=4)
---------------------------
Run from the project root or from this directory. The `--model` argument is the
HF model path; the script infers the model name from the basename to load
`u_corr_*.npy` from `experiments_markerfree/outputs/exp6_3/`.

Example gamma and temperature grids:
- `--gamma_grid "0,2,5,10"`
- `--temp_grid "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0"`

Commands (entropy-matching):

1. Llama-3.1-8B-Instruct

```bash
cd /root/workspace/exp8_temp_baseline
python run_temp_baseline.py \
  --model "/root/workspace/model/meta-llama/Llama-3.1-8B-Instruct" \
  --N 500 \
  --seed 0 \
  --gamma_grid "0,2,5,10" \
  --temp_grid "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0" \
  --match_target "entropy" \
  --out outputs/tempbaseline_Llama-3.1-8B-Instruct.csv
```

2. DeepSeek-R1-Distill-Qwen-7B

```bash
python run_temp_baseline.py \
  --model "/root/workspace/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --N 500 \
  --seed 0 \
  --gamma_grid "0,2,5,10" \
  --temp_grid "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0" \
  --match_target "entropy" \
  --out outputs/tempbaseline_DeepSeek-R1-Distill-Qwen-7B.csv
```

3. Qwen3-8B

```bash
python run_temp_baseline.py \
  --model "/root/workspace/model/Qwen/Qwen3-8B" \
  --N 500 \
  --seed 0 \
  --gamma_grid "0,2,5,10" \
  --temp_grid "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0" \
  --match_target "entropy" \
  --out outputs/tempbaseline_Qwen3-8B.csv
```

4. Qwen2.5-7B-Instruct

```bash
python run_temp_baseline.py \
  --model "/root/workspace/model/Qwen/Qwen2.5-7B-Instruct" \
  --N 500 \
  --seed 0 \
  --gamma_grid "0,2,5,10" \
  --temp_grid "0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0" \
  --match_target "entropy" \
  --out outputs/tempbaseline_Qwen2.5-7B-Instruct.csv
```

After all four runs complete, aggregate and plot:

```bash
python summarize_and_plot.py
```

Expected outputs
----------------
Per model (after `run_temp_baseline.py`):
- `outputs/tempbaseline_<ModelName>.csv`

After `summarize_and_plot.py`:
- `outputs/summary_tempbaseline_all.csv`
- `outputs/fig_proj_vs_tempmatch_accuracy.png`
- `outputs/fig_proj_vs_tempmatch_strictfail.png`
- `outputs/fig_proj_vs_tempmatch_entropy.png`
- `outputs/fig_proj_vs_tempmatch_margin.png`


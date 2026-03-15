# Exp9 Stage-3b: Offline Matched Rank-k Comparison

This folder implements an offline fairness control to compare rank-k projections at matched decision statistics.

No new generations are run. The script only loads existing `rankk_norm_*.csv` outputs and performs piecewise linear interpolation over the gamma grid `[0, 2, 5, 10]`.

## Why this exists

Rank-k (k=4) can change decision confidence differently than k=1 at the same gamma.

To compare k=1 vs k=4 more fairly, we match points where the model has similar:

- entropy over options, or
- margin between top-1 and top-2 option logits

Then we compare accuracy (and optionally failure metrics) at those matched points.

## Usage

From this directory:

```bash
python offline_match_rankk.py --match_target entropy --out outputs/rankk_offline_match_entropy.csv
python offline_match_rankk.py --match_target margin  --out outputs/rankk_offline_match_margin.csv
python summarize_and_plot.py
```

Inputs are expected at either:

- `/mnt/data/rankk_norm_*.csv` (preferred if available), or
- `/root/workspace/exp9_rankk_ablation/outputs/rankk_norm_*.csv` (fallback)

## Outputs

- `outputs/rankk_offline_match_entropy.csv`
- `outputs/rankk_offline_match_margin.csv`
- `outputs/fig_rankk_entropy_matched_acc_delta.png`
- `outputs/fig_rankk_margin_matched_acc_delta.png`
- `outputs/table_rankk_entropy_matched_acc_delta.csv`
- `outputs/table_rankk_margin_matched_acc_delta.csv`

If failure metrics exist in the input CSVs, additional delta plots are also produced.

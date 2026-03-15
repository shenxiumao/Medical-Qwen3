Stage-1 marker-free analysis rebuilt from logged decision-space signals only.

This folder reconstructs the Stage-1 “marker-free” analysis using existing experiment artifacts from `experiments_markerfree/outputs` without running any models or using marker tokens.

Key components:
- `extract_and_index.py`: Copies raw outputs into `outputs_raw/` and builds `outputs_index.json` describing available CSV files and columns.
- `build_alpha_sweep_markerfree.py`: Defines marker-free proxies (Decision Entropy, Choice Margin, Refusal/Format violation) and constructs the Marker-Free Leakage (MFL) metric per `(model, alpha)` condition from logged decision-space signals only.
- `summarize_and_plot.py`: Loads the assembled alpha sweep summary, runs sanity checks, and produces figures comparing marker-based leakage (if available) against MFL, plus entropy and margin trends.

Marker-Free Leakage (MFL)
-------------------------
MFL is a marker-free proxy for leakage that relies purely on decision-space signals:
- Answer entropy (over discrete options)
- Choice margin (difference between top-1 and top-2 option probabilities)
- Logged binary indicators of refusal and format violation

For each model, we identify a baseline condition (e.g., `alpha=0` or `gamma=0`) and derive per-model thresholds:
- `E0`: 75th percentile of entropy under the baseline
- `M0`: 25th percentile of margin under the baseline

For each response, we define a binary indicator:
- High-entropy or low-margin or refusal or format violation:
  `I[entropy > E0 or margin < M0 or format_violation == 1 or refusal_detected == 1]`

The Marker-Free Leakage rate (MFL_rate) is the mean of this indicator in a given condition. By construction:
- MFL does not use any marker strings or text-based pattern matching.
- MFL does not reuse correctness labels (`y_correct`) and is independent of task accuracy.
- MFL is comparable across alpha values within a model via thresholds anchored to that model’s baseline.

Outputs
-------
The pipeline produces:
- `outputs/summary_markerfree_alpha.csv`: Per-model, per-alpha summary with accuracy, MFL_rate, entropy/margin means, and refusal/format violation rates.
- `outputs/fig_alpha_mfl_vs_marker.png`: Alpha vs leakage rates (MFL and marker-hit, if available).
- `outputs/fig_alpha_entropy.png`: Alpha vs mean decision entropy.
- `outputs/fig_alpha_margin.png`: Alpha vs mean choice margin.

No SRL or text-based parsing is used, and no model generations or vLLM calls are required.

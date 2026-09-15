# Battery State-of-Health Estimation from Partial Discharge Cycles

Estimating EV/lithium-ion battery State-of-Health (SoH) using only a truncated
(partial) portion of a discharge cycle, instead of requiring a full discharge — using
a physics-informed neural network with a monotonicity-constrained loss.

## Approach

- **Multi-truncation features**: inputs are built from 30%, 50%, 75%, and 100%
  truncated discharge cycles, so the model can be evaluated on how much of a cycle it
  actually needs to observe.
- **Physics-informed loss**: a regularization term penalizes non-monotonic SoH
  predictions across a battery's cycle life, encoding the physical fact that capacity
  fade is irreversible (SoH should not increase over time for the same cell).
- **Baselines**: a plain LSTM and a full-cycle-only ablation (no partial-cycle
  training) to isolate the contribution of each design choice.
- Validated on the NASA PCoE battery dataset (31 cells). A separate exploratory
  notebook (`battery_soh_transformer.ipynb`) applies a Transformer architecture to the
  larger Severson/MATR (Stanford, 124-cell LFP) dataset.

## Repo layout

- `preprocess_battery.m` — builds the multi-truncation feature set from raw
  discharge-cycle data.
- `train_proposed.m` — trains the physics-informed NN (the proposed model).
- `train_baselines.m`, `train_baseline_lstm.m` — baseline models for comparison.
- `train_proposed_multiseed.m` — multi-seed runs for statistical robustness.
- `ablation_no_physics.m`, `ablation_fullcycle_only.m` — ablations isolating the
  physics loss and the partial-cycle training respectively.
- `eval_per_truncation.m`, `eval_lambda_sensitivity.m` — evaluation across truncation
  levels and physics-loss weight (λ) sensitivity.
- `calc_metrics.m`, `plot_results.m` — shared metrics (RMSE/MAE/MAPE) and plotting.
- `battery_soh_transformer.ipynb` — Transformer-based variant on the Severson dataset.
- `results/` — figures from the reported experiments (loss curves, SoH trajectory
  tracking, predicted-vs-actual scatter, model comparison).
- `data/cleaned_dataset/` — the cleaned NASA PCoE battery discharge-cycle dataset
  (per-cycle CSVs + `metadata.csv`).

## Setup

MATLAB with the Deep Learning Toolbox. The cleaned NASA PCoE dataset is included
directly in this repo at `data/cleaned_dataset/`, so `preprocess_battery.m` can be run
as-is, followed by any of the `train_*.m` / `ablation_*.m` scripts.

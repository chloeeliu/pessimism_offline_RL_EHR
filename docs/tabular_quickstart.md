# Tabular Quickstart

This repository now includes a first tabular-only reproduction scaffold for the offline RL pessimism project.

## What Is Implemented

- a finite-horizon tabular MDP interface
- two built-in tasks:
  - `bandit`: one-step bandit with a weakly covered distractor arm
  - `branching`: three-step branching MDP with an under-covered risky branch
- offline dataset generation from a fixed behavior policy
- empirical model estimation from logged data
- three planners:
  - greedy plug-in value iteration
  - pessimistic value iteration with a count-based penalty
  - support-masked conservative value iteration
- exact policy evaluation on the known simulator
- a sweep script that saves metrics, summaries, and plots

## Main Script

Run experiments with:

```bash
python3 scripts/run_tabular_experiment.py --task branching --dataset-sizes 10,20,50,100 --seeds 80 --beta 0.8
```

Or use the saved study configs:

```bash
python3 scripts/run_tabular_experiment.py --config configs/bandit_study.json
python3 scripts/run_tabular_experiment.py --config configs/branching_study.json
```

Run the beta sweeps with:

```bash
python3 scripts/run_beta_sweep.py --config configs/bandit_beta_sweep.json
python3 scripts/run_beta_sweep.py --config configs/branching_beta_sweep.json
```

Useful options:

- `--task bandit`
- `--task branching`
- `--dataset-sizes 10,20,50,100,200`
- `--seeds 50`
- `--beta 0.8`
- `--support-mask-threshold 2`
- `--config configs/branching_study.json`
- `--output-dir results/raw/my_run`

## Outputs

Each run writes:

- `metrics.csv`: one row per method, seed, and dataset size
- `summary.json`: aggregated metrics by method and dataset size
- `resolved_config.json`: exact settings used for the run
- `comparison.png`: return and low-support plots

Beta sweep runs write:

- `metrics.csv`
- `summary.json`
- `resolved_config.json`
- `beta_sweep_values.png`
- `beta_sweep_diagnostics.png`

Default output directory:

```text
results/raw/<task>/
```

## Current Interpretation

The implementation is intentionally simple and is meant to validate the mechanism first:

- the greedy plug-in baseline can over-commit to weakly supported actions
- the pessimistic planner subtracts a count-based uncertainty penalty
- the comparison is exact because evaluation happens on the known tabular environment

This is not yet a theorem-faithful linear MDP implementation and it does not use deep RL.

Current study docs:

- [phase1_tabular_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/phase1_tabular_report.md)
- [beta_sweep_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/beta_sweep_report.md)

## Suggested Next Coding Steps

1. Add harder synthetic tasks where missing coverage creates genuine intrinsic uncertainty.
2. Log more per-step diagnostics such as chosen bonuses and root-action statistics in the plots.
3. Try an adaptive or data-dependent pessimism schedule instead of a fixed `beta`.
4. Add one synthetic linear-feature task after the tabular story is stable.

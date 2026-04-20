# Linear Quickstart

This project now has a working Phase 2 linear-feature experiment suite alongside the tabular suite.

## What Exists Now

- a linear task package in [src/peorl/linear](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/src/peorl/linear)
- four synthetic task builders:
  - `linear_bandit`
  - `linear_branching`
  - `linear_intrinsic`
  - `linear_near_intrinsic`
- linear dataset collection utilities
- ridge-regression and confidence-width helpers
- fixed-beta linear study runners and beta sweeps
- generated reports and plots for the first linear studies

## Main Design Note

- [phase2_linear_design.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/phase2_linear_design.md)
- [phase2_linear_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/phase2_linear_report.md)
- [linear_beta_sweep_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/linear_beta_sweep_report.md)
- [four_task_detailed_guide.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/four_task_detailed_guide.md)
- [cross_suite_analysis_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/cross_suite_analysis_report.md)

## Useful Commands

Inspect the current linear tasks:

```bash
python3 scripts/inspect_linear_tasks.py --task linear_bandit --feature-dim 6
python3 scripts/inspect_linear_tasks.py --task linear_branching --feature-dim 6
```

Run the fixed-beta linear studies:

```bash
python3 scripts/run_linear_experiment.py --config configs/linear_bandit_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_branching_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_intrinsic_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_near_intrinsic_study.json
```

Run the linear beta sweeps:

```bash
python3 scripts/run_linear_beta_sweep.py --config configs/linear_bandit_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_branching_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_intrinsic_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_near_intrinsic_beta_sweep.json
```

## Current Status

Implemented:

- linear task dataclasses
- synthetic feature tensors
- offline dataset collection
- ridge solve and confidence width utilities
- finite-horizon linear PEVI
- fixed-beta study runner
- beta sweep runner
- linear reports with saved plots and metrics

Current study outputs:

- [linear bandit study](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study)
- [linear branching study](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study)
- [linear intrinsic study](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study)
- [linear near-intrinsic study](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study)
- [linear bandit beta sweep](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep)
- [linear branching beta sweep](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep)
- [linear intrinsic beta sweep](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep)
- [linear near-intrinsic beta sweep](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep)

## Recommended Next Step

Use the completed boundary-study suite as the bridge toward rolling EHR state representations:

- keep `linear_intrinsic` as the off-support sanity check
- keep `linear_near_intrinsic` as the rare-support turnover check
- then replace synthetic features with rolling clinical features while preserving the same diagnostics

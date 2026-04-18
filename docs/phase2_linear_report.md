# Phase 2 Linear Report

## Scope

This report documents the first linear-feature experiments for the offline RL pessimism project.

The aim of this phase is narrower than the full paper:

- keep synthetic control and exact evaluation
- replace tabular value lookup with shared linear features
- test whether pessimism still helps under feature-based generalization

## Assets

Code:

- [src/peorl/linear](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/src/peorl/linear)
- [scripts/run_linear_experiment.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/scripts/run_linear_experiment.py)
- [scripts/run_linear_beta_sweep.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/scripts/run_linear_beta_sweep.py)

Configs:

- [configs/linear_bandit_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_bandit_study.json)
- [configs/linear_branching_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_branching_study.json)
- [configs/linear_intrinsic_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_intrinsic_study.json)
- [configs/linear_near_intrinsic_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_near_intrinsic_study.json)

## Tasks

### 1. Linear Bandit

- one-step linear-feature bandit
- weakly covered distractor action
- shared feature representation across actions

### 2. Linear Branching MDP

- three-step linear-feature planning task
- safe branch is supported
- hidden-optimal path is weakly represented in the behavior policy

## Methods

- `greedy`: linear Bellman regression with no pessimistic penalty
- `pessimistic`: linear Bellman regression with covariance-based penalty
- `support_masked`: conservative ablation using empirical support thresholding on the discrete simulator

## Results
The full fixed-beta runs were executed with:

```bash
python3 scripts/run_linear_experiment.py --config configs/linear_bandit_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_branching_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_intrinsic_study.json
python3 scripts/run_linear_experiment.py --config configs/linear_near_intrinsic_study.json
```

Common settings:

- `feature_dim = 6`
- dataset sizes: `10, 20, 50, 100, 200, 500`
- seeds per point: `200`
- linear ridge parameter: `1.0`
- pessimistic coefficient for the fixed study: `beta = 0.8`

### 1. Linear Bandit

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study/metrics.csv)
- [comparison plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study/comparison.png)

Reference values:

- optimal value: `0.780`
- behavior policy value: `0.565`

Key outcomes:

- `10` episodes:
  - greedy: `0.664 +/- 0.126`
  - pessimistic: `0.580 +/- 0.171`
  - support-masked: `0.708 +/- 0.143`
- `50` episodes:
  - greedy: `0.699 +/- 0.130`
  - pessimistic: `0.718 +/- 0.130`
  - support-masked: `0.711 +/- 0.127`
- `100` episodes:
  - greedy: `0.752 +/- 0.088`
  - pessimistic: `0.761 +/- 0.077`
  - support-masked: `0.752 +/- 0.088`
- `500` episodes:
  - all methods converge to `0.780`

Diagnostic trends:

- greedy places substantial mass on low-support actions at tiny data:
  - `0.535` at `10`
  - `0.345` at `20`
  - `0.070` at `50`
- pessimism lowers that mass more slowly than the hard support mask at very small sample sizes, but eventually improves value in the middle regime:
  - low-support mass `0.020` at `50`
  - action-agreement mass `0.815` at `50`
- the chosen-feature novelty under pessimism decays as coverage grows:
  - `0.282` at `10`
  - `0.158` at `50`
  - `0.053` at `500`

Visualization:

![Linear bandit fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_study/comparison.png)

Interpretation:

- The linear bandit preserves the basic weak-coverage story, but it is less clean than the tabular bandit.
- A fixed `beta = 0.8` is too conservative in the tiniest-data regime, mildly helpful in the middle regime, and irrelevant once coverage becomes adequate.
- The support-mask ablation is stronger at `10` and `20`, which suggests that in this simple shared-feature setting, hard empirical filtering can outperform a moderate covariance bonus when data is extremely scarce.

### 2. Linear Branching MDP

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study/metrics.csv)
- [comparison plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study/comparison.png)

Reference values:

- optimal value: `0.950`
- behavior policy value: `0.743`

Key outcomes:

- `10` episodes:
  - greedy: `0.762 +/- 0.121`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.761 +/- 0.083`
- `50` episodes:
  - greedy: `0.735 +/- 0.083`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.731 +/- 0.077`
- `100` episodes:
  - greedy: `0.743 +/- 0.063`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.743 +/- 0.063`
- `500` episodes:
  - greedy: `0.767 +/- 0.047`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.767 +/- 0.047`

Diagnostic trends:

- greedy starts with very high low-support mass:
  - `0.900` at `10`
  - `0.525` at `20`
  - `0.090` at `50`
- pessimism drives low-support mass to `0.000` immediately at every dataset size.
- pessimism is deliberately conservative in value estimation:
  - chosen-action q-error `-0.360` at `10`
  - `-0.208` at `50`
  - `-0.058` at `500`
- action-agreement mass under pessimism stays near `0.667`, matching the safe-branch policy rather than the hidden optimal path.

Visualization:

![Linear branching fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_study/comparison.png)

### 3. Linear Intrinsic Off-Support Task

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study/metrics.csv)
- [comparison plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study/comparison.png)

Reference values:

- optimal value: `0.950`
- safe supported value: `0.800`

Task-specific setting:

- `feature_dim = 8`
- root risky probability in behavior: `0.04`
- hidden optimal continuation probability in behavior: `0.00`

Key outcomes:

- `10` episodes:
  - greedy: `0.745 +/- 0.142`
  - pessimistic: `0.789 +/- 0.070`
  - support-masked: `0.764 +/- 0.089`
- `50` episodes:
  - greedy: `0.735 +/- 0.116`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.742 +/- 0.080`
- `200` episodes:
  - greedy: `0.777 +/- 0.085`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.756 +/- 0.055`
- `500` episodes:
  - greedy: `0.812 +/- 0.082`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.775 +/- 0.044`

Diagnostic trends:

- greedy remains exposed to low-support actions even at high coverage:
  - `0.800` low-support mass at `10`
  - `0.315` at `50`
  - `0.430` at `500`
- pessimism collapses onto the supported safe policy by `50` episodes and stays there.
- unlike the tabular intrinsic case, the greedy linear estimator leaks slightly above `0.800` at large data because function approximation extrapolates across features, but it still remains far below the true hidden optimum `0.950`.

Visualization:

![Linear intrinsic fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_study/comparison.png)

Interpretation:

- This is the clearest linear analogue of graceful failure.
- Pessimism reliably secures the best clearly supported value.
- The hidden optimum is still not recovered; instead, the function approximator produces mild upward leakage rather than true recovery.

### 4. Linear Near-Intrinsic Rare-Support Task

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study/metrics.csv)
- [comparison plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study/comparison.png)

Reference values:

- optimal value: `0.950`
- safe supported value: `0.800`

Task-specific setting:

- `feature_dim = 8`
- root risky probability in behavior: `0.10`
- hidden optimal continuation probability in behavior: `0.10`

Key outcomes:

- `10` episodes:
  - greedy: `0.760 +/- 0.154`
  - pessimistic: `0.789 +/- 0.070`
  - support-masked: `0.760 +/- 0.095`
- `50` episodes:
  - greedy: `0.744 +/- 0.143`
  - pessimistic: `0.798 +/- 0.032`
  - support-masked: `0.730 +/- 0.107`
- `100` episodes:
  - greedy: `0.796 +/- 0.101`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.770 +/- 0.090`
- `200` episodes:
  - greedy: `0.828 +/- 0.110`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.817 +/- 0.106`
- `500` episodes:
  - greedy: `0.894 +/- 0.087`
  - pessimistic: `0.800 +/- 0.000`
  - support-masked: `0.893 +/- 0.088`

Diagnostic trends:

- early on, pessimism is still useful because it kills low-support action mass:
  - greedy low-support mass `1.015` at `10`
  - pessimistic low-support mass `0.025` at `10`
- by `200` and `500` episodes, greedy and support-masked have enough rare evidence to move past the safe value, while fixed pessimism stays pinned to `0.800`.
- action-agreement mass for greedy grows from `0.692` at `10` to `0.897` at `500`, showing real recovery toward the hidden optimal policy.

Visualization:

![Linear near-intrinsic fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_study/comparison.png)

Interpretation:

- This task gives the linear version of the tabular turnover story.
- A fixed positive `beta` helps in the weak-data regime, but eventually becomes too conservative once rare support accumulates.
- The support-masked baseline follows the same high-coverage recovery pattern as greedy, which reinforces that the key issue is no longer absence of evidence, but over-conservatism.

## Interpretation
This first linear phase is already informative:

- The shared-feature setting does preserve the main qualitative idea from tabular: pessimism can still suppress weakly supported actions and improve policy value.
- But the linear bandit is much more beta-sensitive than the tabular bandit, which is a useful warning for the EHR direction: under function approximation, conservatism depends much more on representation and tuning.
- The original linear branching task behaves like a feature-based analogue of the intrinsic support-limit case. Pessimism reliably reaches the best supported value `0.800`, but it does not recover the hidden optimal value `0.950`.
- The new boundary tasks make the story much sharper:
  - `linear_intrinsic`: the hidden continuation is absent, so pessimism protects the safe value and no method approaches the true optimum
  - `linear_near_intrinsic`: the hidden continuation is rare, so fixed pessimism helps early and hurts later as rare evidence accumulates

The main takeaway is that moving beyond tabular was the right next step, and the richer linear family is now doing useful work. The next improvement is to push this same logic into a more EHR-like setting with:

- higher-dimensional rolling features
- more realistic overlap and misspecification
- a clearer distinction between support failure and representation failure

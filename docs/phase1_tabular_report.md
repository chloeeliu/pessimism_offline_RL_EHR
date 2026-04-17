# Phase 1 Tabular Report

## Scope

This report documents the first tabular-only reproduction pass for the project based on *Is Pessimism Provably Efficient for Offline RL?*.

Phase 1 focuses on a simple but testable question:

- when offline data has weak support, does a pessimistic planner avoid unsupported optimistic actions more reliably than a greedy plug-in baseline?

The implementation in this phase is intentionally lightweight:

- finite-horizon tabular MDPs
- offline data generated from fixed behavior policies
- exact evaluation on the known simulator
- no function approximation and no deep RL

## Experiment Assets

Code:

- [scripts/run_tabular_experiment.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/scripts/run_tabular_experiment.py)
- [src/peorl/envs/tabular.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/src/peorl/envs/tabular.py)
- [src/peorl/algorithms/tabular_vi.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/src/peorl/algorithms/tabular_vi.py)
- [src/peorl/experiments.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/src/peorl/experiments.py)

Configs:

- [configs/bandit_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/bandit_study.json)
- [configs/branching_study.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/branching_study.json)
- [configs/bandit_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/bandit_beta_sweep.json)
- [configs/branching_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/branching_beta_sweep.json)

## Tasks

### 1. Spurious Bandit

- one-state, one-step bandit
- one truly best arm
- one weakly covered distractor arm with high but suboptimal reward
- behavior policy samples the distractor rarely

Purpose:

- isolate spurious correlation in the smallest possible setting

### 2. Branching MDP

- three-step episodic MDP
- safe branch is well covered and truly optimal
- risky branch is weakly covered and can look attractive under noisy empirical estimates

Purpose:

- test whether pessimism improves end-to-end offline planning, not just one-step action selection

## Methods

- `greedy`: plug-in value iteration with no uncertainty penalty
- `pessimistic`: plug-in value iteration minus a count-based bonus proportional to `1 / sqrt(n)`
- `support_masked`: conservative ablation that forbids actions below a support threshold when any supported action exists

## Diagnostics

In addition to exact policy value, the study logs:

- occupancy mass on low-support actions
- occupancy-weighted chosen-action overestimation, `q_hat - q_true`
- occupancy-weighted agreement with the optimal policy
- chosen root action, root support count, and root penalty

## Results

Commands used:

```bash
python3 scripts/run_tabular_experiment.py --config configs/bandit_study.json
python3 scripts/run_tabular_experiment.py --config configs/branching_study.json
```

Common settings:

- dataset sizes: `10, 20, 50, 100, 200, 500`
- seeds: `200`
- pessimism coefficient: `beta = 0.8`
- support-masked threshold: `3` for the bandit, `2` for the branching MDP

Each run also saves the resolved config:

- [bandit resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/resolved_config.json)
- [branching resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/resolved_config.json)

### Bandit

Artifacts:

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/metrics.csv)
- [comparison.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/comparison.png)

Figure:

![Bandit fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_study/comparison.png)

Key findings:

- At `10` episodes, greedy reached mean value `0.619`, pessimistic reached `0.700`, and support-masked reached `0.725`.
- At `20` episodes, greedy dropped to `0.564`, while pessimistic and support-masked stayed near `0.708` and `0.710`.
- At `50` episodes, pessimistic remained clearly ahead at `0.743` versus greedy `0.602` and support-masked `0.650`.
- By `500` episodes, all methods had mostly converged: greedy `0.775`, pessimistic `0.779`, support-masked `0.775`.

Diagnostics:

- Greedy put heavy mass on low-support actions in the hardest regimes: `0.585` at `10` episodes and `0.580` at `20`.
- Pessimistic sharply reduced that mass to `0.285` at `10` and `0.075` at `20`.
- Greedy’s chosen-action overestimation was large early: `0.327` at `10` and `0.384` at `20`.
- Pessimistic cut that to `0.162` at `10` and `0.129` at `20`.
- Agreement with the optimal policy was much higher for pessimism in the low-data regime: `0.760` versus greedy `0.515` at `10`, and `0.800` versus `0.360` at `20`.

Interpretation:

- The bandit reproduces the intended mechanism cleanly.
- Greedy often chases weakly sampled arms when the dataset is tiny.
- Pessimism reduces both overestimation and low-support action selection.
- The support-masked baseline is very competitive at `10` and `20` episodes, but it degrades noticeably at `50` and `100` episodes because hard thresholding can remain too blunt after the most dangerous low-support actions have disappeared.

### Branching

Artifacts:

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/metrics.csv)
- [comparison.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/comparison.png)

Figure:

![Branching fixed-beta comparison](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_study/comparison.png)

Key findings:

- At `10` episodes, greedy reached mean value `0.801`, pessimistic reached `0.898`, and support-masked reached `0.876`.
- At `20` episodes, greedy fell to `0.757`, pessimistic stayed at `0.896`, and support-masked reached `0.851`.
- At `50` episodes, pessimistic remained near-optimal at `0.896`, while greedy was still only `0.787`.
- By `200` episodes, all methods were close, but pessimistic had already essentially converged to `0.900`.

Diagnostics:

- Greedy still placed substantial mass on low-support actions at `10` and `20` episodes: `0.405` and `0.395`.
- Pessimistic nearly eliminated that immediately: `0.005` and `0.010`.
- Greedy’s chosen-action overestimation was positive and persistent early: `0.147`, `0.186`, `0.119`.
- Pessimistic was deliberately conservative, with negative chosen-action error: `-0.337`, `-0.251`, `-0.166`.
- Agreement with the optimal policy was already high for greedy because most states stay on the safe branch, but pessimism still improved it from `0.917` to `0.998` at `10` episodes and from `0.883` to `0.997` at `20`.

Root-action behavior at `10` episodes:

- Greedy chose the risky root action in `35 / 200` seeds.
- Pessimistic chose the safe root action in `200 / 200` seeds.
- Support-masked still chose the risky root action in `11 / 200` seeds.

Interpretation:

- The branching task is the stronger end-to-end planning demonstration.
- Pessimism does not just prune rare actions; it keeps the planner on the correct branch across multiple steps.
- The support-masked baseline helps, but it is less consistent because a simple support threshold cannot express graded uncertainty across the horizon.

## Interpretation

The first-phase results support the design choice to stay tabular before moving to richer function classes.

What worked:

- The main qualitative claim from the paper shows up in both tasks.
- The greedy plug-in baseline is most fragile when support is weakest.
- Pessimism improves return precisely by reducing reliance on poorly supported actions and lowering chosen-action overestimation.
- As dataset size grows, the difference between methods shrinks, which is the behavior we wanted to see.

What the extra baseline taught us:

- A simple conservative rule can help a lot in the smallest-data regime.
- That same rule can become too rigid in the medium-data regime, especially in the bandit, where support masking lagged behind the pessimistic method once more nuanced distinctions mattered.
- This makes the report stronger, because it shows the advantage is not just “being conservative,” but using a graded uncertainty penalty.

What still needs caution:

- The count-based penalty here is a practical tabular surrogate, not the paper’s exact theoretical object.
- The tasks were designed to expose failure modes clearly, so they are diagnostic environments, not a substitute for benchmark evidence.
- Negative chosen-action error for the pessimistic method in the branching task is expected from an intentionally conservative planner, but it also reminds us that the current penalty may be stronger than necessary.

## Beta Sweep Update

The fixed-`beta` story is now complemented by a dedicated sweep report:

- [docs/beta_sweep_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/beta_sweep_report.md)

Latest sweep takeaway:

- the bandit prefers fairly strong pessimism across the tested range
- the branching task prefers moderate pessimism under weak coverage and progressively smaller `beta` as coverage improves
- `beta = 0.8` remains a good default compromise for this repository, but it is not universally optimal

## Design Takeaways for Phase 2

The tabular results suggest the next extension should preserve three properties:

- keep coverage measurable and explicit
- keep exact or near-exact evaluation available when possible
- keep uncertainty penalties inspectable rather than burying them inside a black-box optimizer

Recommended next coding step:

- add a `beta` sweep and identify the range where pessimism helps without becoming too conservative

Recommended next scientific step:

- create one or two harder branching families where intrinsic uncertainty is genuinely unrecoverable, so we can show pessimism fails gracefully rather than simply looking strong everywhere

## Limits

- this is a tabular reproduction scaffold, not a faithful linear-MDP theorem implementation
- the pessimistic penalty is intentionally simple and count-based
- synthetic tasks make diagnosis easier, but they are not yet benchmark-level evidence

## Next Steps

- add parameter sweeps over the pessimism coefficient `beta`
- compare against a fixed-penalty conservative baseline in addition to support masking
- add per-step plots of estimated Q, true Q, and penalty size on chosen trajectories
- move to a linear-feature version only after the tabular picture is stable

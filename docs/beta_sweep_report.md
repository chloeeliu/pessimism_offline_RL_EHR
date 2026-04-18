# Beta Sweep Report

## Goal

This report studies how sensitive the tabular reproduction is to the pessimism coefficient `beta`.

The main questions are:

- how much pessimism is enough to suppress spurious correlation?
- when does extra pessimism stop helping?
- does the preferred `beta` depend on the task and the dataset size?

## Sweep Setup

Commands:

```bash
python3 scripts/run_beta_sweep.py --config configs/bandit_beta_sweep.json
python3 scripts/run_beta_sweep.py --config configs/branching_beta_sweep.json
python3 scripts/run_beta_sweep.py --config configs/intrinsic_beta_sweep.json
python3 scripts/run_beta_sweep.py --config configs/near_intrinsic_beta_sweep.json
```

Sweep values:

- `beta in {0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0}`

Common settings:

- dataset sizes: `10, 20, 50, 100, 200, 500`
- seeds: `200`
- support-masked baseline retained for reference

Code and configs:

- [scripts/run_beta_sweep.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/scripts/run_beta_sweep.py)
- [configs/bandit_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/bandit_beta_sweep.json)
- [configs/branching_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/branching_beta_sweep.json)
- [configs/intrinsic_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/intrinsic_beta_sweep.json)
- [configs/near_intrinsic_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/near_intrinsic_beta_sweep.json)

## Artifacts

### Bandit

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/metrics.csv)
- [beta_sweep_values.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/beta_sweep_values.png)
- [beta_sweep_diagnostics.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/beta_sweep_diagnostics.png)

### Branching

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/metrics.csv)
- [beta_sweep_values.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/beta_sweep_values.png)
- [beta_sweep_diagnostics.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/beta_sweep_diagnostics.png)

### Intrinsic Uncertainty

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/metrics.csv)
- [beta_sweep_values.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/beta_sweep_values.png)
- [beta_sweep_diagnostics.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/beta_sweep_diagnostics.png)

### Near-Missing Support

- [resolved_config.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/resolved_config.json)
- [summary.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/summary.json)
- [metrics.csv](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/metrics.csv)
- [beta_sweep_values.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/beta_sweep_values.png)
- [beta_sweep_diagnostics.png](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

## Results

### Bandit

Best-by-size summary:

- `10` episodes: best `beta = 2.0`, mean value `0.780`, improvement over greedy `+0.161`
- `20` episodes: best `beta = 2.0`, mean value `0.780`, improvement over greedy `+0.216`
- `50` episodes: best `beta = 2.0`, mean value `0.780`, improvement over greedy `+0.178`
- `100` episodes: best `beta = 1.6`, mean value `0.780`, improvement over greedy `+0.080`
- `200` episodes: best `beta = 2.0`, mean value `0.780`, improvement over greedy `+0.033`
- `500` episodes: best `beta = 1.6`, mean value `0.780`, improvement over greedy `+0.005`

Representative values:

- At `10` episodes, greedy was `0.619`, `beta = 0.8` reached `0.700`, `beta = 1.2` reached `0.736`, and `beta = 2.0` reached `0.780`.
- At `20` episodes, greedy was `0.564`, `beta = 0.8` reached `0.708`, `beta = 1.2` reached `0.742`, and `beta = 2.0` again reached `0.780`.
- Even at `500` episodes, the sweep still preferred a nonzero penalty, though the practical gain was tiny: `0.780` versus greedy `0.775`.

Diagnostic pattern:

- Larger `beta` monotonically reduced chosen-action overestimation in the bandit.
- The best settings pushed chosen-action Q error slightly below zero, which is consistent with mild intentional conservatism.
- Low-support mass was driven almost to zero by moderate-to-large `beta`.

Figures:

![Bandit beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/beta_sweep_values.png)

![Bandit beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/bandit_beta_sweep/beta_sweep_diagnostics.png)

### Branching

Best-by-size summary:

- `10` episodes: best `beta = 1.2`, mean value `0.900`, improvement over greedy `+0.099`
- `20` episodes: best `beta = 1.2`, mean value `0.900`, improvement over greedy `+0.143`
- `50` episodes: best `beta = 1.2`, mean value `0.900`, improvement over greedy `+0.113`
- `100` episodes: best `beta = 0.8`, mean value `0.900`, improvement over greedy `+0.065`
- `200` episodes: best `beta = 0.2`, mean value `0.900`, improvement over greedy `+0.011`
- `500` episodes: best `beta = 0.0`, mean value `0.900`, improvement over greedy `+0.000`

Representative values:

- At `10` episodes, greedy was `0.801`, `beta = 0.8` reached `0.898`, and both `beta = 1.2` and `beta = 2.0` reached `0.900`.
- At `20` episodes, greedy was `0.757`, `beta = 0.8` reached `0.896`, and `beta = 1.2` reached `0.900`.
- At `200` episodes, almost any positive `beta` already reached the optimum, but the selected best beta had dropped to `0.2`.
- At `500` episodes, the best setting was `beta = 0.0`, meaning the task no longer needed pessimism once coverage was abundant.

Diagnostic pattern:

- The branching task preferred moderate pessimism when coverage was weak, not maximal pessimism.
- As coverage improved, the preferred `beta` decreased smoothly toward zero.
- Stronger `beta` values still achieved optimal value in some medium-to-high coverage regimes, but they did so with more negative chosen-action Q error than necessary.

Figures:

![Branching beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/beta_sweep_values.png)

![Branching beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/branching_beta_sweep/beta_sweep_diagnostics.png)

### Intrinsic Uncertainty

Best-by-size summary:

- `10` episodes: best `beta = 1.2`, mean value `0.800`, improvement over greedy `+0.061`
- `20` episodes: best `beta = 1.2`, mean value `0.800`, improvement over greedy `+0.082`
- `50` episodes: best `beta = 0.8`, mean value `0.800`, improvement over greedy `+0.065`
- `100` episodes: best `beta = 0.4`, mean value `0.800`, improvement over greedy `+0.045`
- `200` episodes: best `beta = 0.4`, mean value `0.800`, improvement over greedy `+0.027`
- `500` episodes: best `beta = 0.8`, mean value `0.800`, improvement over greedy `+0.018`

Representative values:

- The true optimum in this task is `0.950`, but no tested `beta` exceeds `0.800`.
- At `10` episodes, greedy was `0.738`, support-masked was `0.780`, `beta = 0.8` reached `0.786`, and `beta = 1.2` reached `0.800`.
- At `20` episodes, greedy was `0.718`, `beta = 0.4` reached `0.781`, `beta = 0.8` reached `0.791`, and `beta = 1.2` reached `0.800`.
- From `100` episodes onward, a broad range of positive betas already reached the same ceiling value `0.800`.

Diagnostic pattern:

- Positive `beta` values rapidly eliminated low-support mass, but they did not create any path to the hidden optimum.
- Once the learner committed to the safe supported branch, stronger `beta` mostly changed the degree of conservatism, not the achieved value.
- The best-beta curve is comparatively flat because the ceiling is imposed by missing support, not just by overestimation.

Figures:

![Intrinsic beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/beta_sweep_values.png)

![Intrinsic beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/intrinsic_beta_sweep/beta_sweep_diagnostics.png)

### Near-Missing Support

Best-by-size summary:

- `10` episodes: best `beta = 1.2`, mean value `0.800`, improvement over greedy `+0.054`
- `20` episodes: best `beta = 1.2`, mean value `0.800`, improvement over greedy `+0.060`
- `50` episodes: best `beta = 1.2`, mean value `0.800`, improvement over greedy `+0.009`
- `100` episodes: best `beta = 0.0`, mean value `0.846`, improvement over greedy `+0.000`
- `200` episodes: best `beta = 0.0`, mean value `0.887`, improvement over greedy `+0.000`
- `500` episodes: best `beta = 0.0`, mean value `0.911`, improvement over greedy `+0.000`

Representative values:

- At `10` episodes, greedy was `0.746`, `beta = 0.8` reached `0.786`, and `beta = 1.2` reached `0.800`.
- At `50` episodes, greedy had already climbed to `0.791`, while `beta = 1.2` stayed at `0.800`.
- At `100` episodes, the best setting was already `beta = 0.0`, with mean value `0.846`, while `beta = 0.8` remained stuck near `0.798`.
- At `500` episodes, `beta = 0.0` reached `0.911`, while `beta = 0.4` only reached `0.803` and larger betas stayed effectively pinned at `0.800`.

Diagnostic pattern:

- Positive `beta` values erased low-support mass very quickly, but they also blocked the planner from taking advantage of the rare hidden-support evidence.
- The best-beta curve changes abruptly rather than smoothly: strong pessimism is best early, but by `100` episodes the optimal setting has already dropped to no pessimism.
- This task makes the cost of fixed pessimism concrete: once rare support becomes statistically meaningful, continuing to penalize it can suppress real signal rather than just noise.

Figures:

![Near-intrinsic beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/beta_sweep_values.png)

![Near-intrinsic beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/near_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

## Interpretation

The sweep adds an important nuance to the phase-1 story.

What looks stable across tasks:

- `beta = 0.0` behaves exactly like the greedy baseline, as expected.
- Some positive pessimism is consistently helpful in the low-coverage regime.
- The main mechanism still tracks the paper’s story: increasing pessimism reduces overestimation and suppresses unsupported action choices.

What differs by task:

- The bandit behaves like a pure spurious-correlation problem, so stronger pessimism keeps helping across almost the entire tested range.
- The branching task behaves more like a planning problem with a meaningful safe path already present in the data, so moderate pessimism is enough early, and the best `beta` shrinks as coverage improves.
- The intrinsic task behaves like a true support limit: tuning `beta` improves robustness, but it cannot move the learner past the best value supported by the dataset.
- The near-support task sits between those cases: early on it looks like the intrinsic task, but later it flips into a regime where the hidden branch is learnable and any fixed positive `beta` becomes too conservative.

Practical takeaway:

- For the current tabular tasks, `beta = 0.8` was a good default, but it was not uniformly best.
- If we care about robust low-data behavior without task-specific tuning, `beta` in roughly the `0.8` to `1.2` range looks like a strong compromise.
- If we later move to harder synthetic tasks or EHR-like settings, a sweep or adaptive rule for the penalty strength will probably matter.
- In the bandit, the best setting often hit the top of the tested range, so that result should be read as “strong pessimism is helpful here,” not as proof that `2.0` is a universal optimum.
- In the intrinsic task, the sweep should be read as a robustness study, not an optimization study: the main result is that no `beta` can overcome absent support.
- In the near-support task, the main result is that support quality and penalty strength interact dynamically; this is the clearest evidence so far that a fixed pessimism coefficient is not enough if support changes from effectively absent to weakly informative.

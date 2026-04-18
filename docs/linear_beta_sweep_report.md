# Linear Beta Sweep Report

## Scope

This report studies how the linear-feature experiments depend on the pessimism coefficient `beta`.

## Assets

- [configs/linear_bandit_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_bandit_beta_sweep.json)
- [configs/linear_branching_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_branching_beta_sweep.json)
- [configs/linear_intrinsic_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_intrinsic_beta_sweep.json)
- [configs/linear_near_intrinsic_beta_sweep.json](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/configs/linear_near_intrinsic_beta_sweep.json)
- [scripts/run_linear_beta_sweep.py](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/scripts/run_linear_beta_sweep.py)

## Results
The full linear beta sweeps were executed with:

```bash
python3 scripts/run_linear_beta_sweep.py --config configs/linear_bandit_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_branching_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_intrinsic_beta_sweep.json
python3 scripts/run_linear_beta_sweep.py --config configs/linear_near_intrinsic_beta_sweep.json
```

Sweep values:

- `beta in {0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0}`

### 1. Linear Bandit

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/metrics.csv)
- [value plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/beta_sweep_values.png)
- [diagnostic plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/beta_sweep_diagnostics.png)

Best beta by dataset size:

- `10`: best `beta = 2.0`, value `0.707`, improvement over greedy `+0.043`
- `20`: best `beta = 0.4`, value `0.665`, improvement `+0.002`
- `50`: best `beta = 0.8`, value `0.718`, improvement `+0.020`
- `100`: best `beta = 0.8`, value `0.761`, improvement `+0.009`
- `200`: best `beta = 0.8`, value `0.777`, improvement `+0.003`
- `500`: best `beta = 0.0`, value `0.780`, improvement `+0.000`

Representative comparisons:

- `10` episodes:
  - greedy: `0.664`
  - support-masked: `0.708`
  - `beta=0.8`: `0.580`
  - `beta=2.0`: `0.707`
- `20` episodes:
  - greedy: `0.663`
  - support-masked: `0.689`
  - `beta=0.4`: `0.665`
  - `beta=2.0`: `0.568`
- `50` episodes:
  - greedy: `0.699`
  - support-masked: `0.711`
  - `beta=0.4`: `0.714`
  - `beta=0.8`: `0.718`
- `100` episodes:
  - greedy: `0.752`
  - `beta=0.8`: `0.761`
  - `beta=2.0`: `0.709`

Interpretation:

- The linear bandit has a real coverage effect, but it is non-monotone in `beta`.
- A moderate penalty is best in the middle regime, while a very strong penalty is only helpful at the tiniest dataset size.
- This is noticeably more delicate than the tabular case and suggests that feature sharing changes the effective conservatism induced by the same nominal `beta`.

Visualization:

![Linear bandit beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/beta_sweep_values.png)

![Linear bandit beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_bandit_beta_sweep/beta_sweep_diagnostics.png)

### 2. Linear Branching MDP

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/metrics.csv)
- [value plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/beta_sweep_values.png)
- [diagnostic plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/beta_sweep_diagnostics.png)

Best beta by dataset size:

- `10`: best `beta = 0.8`, value `0.800`, improvement over greedy `+0.038`
- `20`: best `beta = 0.8`, value `0.800`, improvement `+0.060`
- `50`: best `beta = 0.8`, value `0.800`, improvement `+0.065`
- `100`: best `beta = 0.8`, value `0.800`, improvement `+0.057`
- `200`: best `beta = 0.8`, value `0.800`, improvement `+0.042`
- `500`: best `beta = 0.8`, value `0.800`, improvement `+0.033`

Representative comparisons:

- `10` episodes:
  - greedy: `0.762`
  - support-masked: `0.761`
  - `beta=0.4`: `0.785`
  - `beta=0.8`: `0.800`
  - `beta=2.0`: `0.800`
- `50` episodes:
  - greedy: `0.735`
  - support-masked: `0.731`
  - `beta=0.4`: `0.798`
  - `beta=0.8`: `0.800`
  - `beta=2.0`: `0.800`
- `500` episodes:
  - greedy: `0.767`
  - support-masked: `0.767`
  - `beta=0.4`: `0.800`
  - `beta=0.8`: `0.800`
  - `beta=2.0`: `0.800`

Interpretation:

- The current linear branching task is much more stable than the bandit under `beta` tuning.
- Once `beta` is large enough to keep the policy on the supported safe branch, extra pessimism changes the diagnostics but not the policy value.
- The stable plateau at `0.800` confirms that this task is really testing a support limit, not recovery of the hidden optimal branch.

Visualization:

![Linear branching beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/beta_sweep_values.png)

![Linear branching beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_branching_beta_sweep/beta_sweep_diagnostics.png)

### 3. Linear Intrinsic Off-Support Task

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/metrics.csv)
- [value plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/beta_sweep_values.png)
- [diagnostic plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

Best beta by dataset size:

- `10`: best `beta = 1.2`, value `0.800`, improvement over greedy `+0.055`
- `20`: best `beta = 1.2`, value `0.800`, improvement `+0.070`
- `50`: best `beta = 0.8`, value `0.800`, improvement `+0.065`
- `100`: best `beta = 0.8`, value `0.800`, improvement `+0.035`
- `200`: best `beta = 0.4`, value `0.800`, improvement `+0.023`
- `500`: best `beta = 0.0`, value `0.812`, improvement `+0.000`

Interpretation:

- The preferred `beta` shrinks as coverage grows, but the best achievable value still stays far below the hidden optimum `0.950`.
- The `500`-episode crossover to `beta=0.0` is not true recovery of the off-support optimum; it is mild linear extrapolation leakage above the safe `0.800` plateau.
- This makes the linear off-support case more nuanced than the tabular one, but the core lesson survives: pessimism protects against unsupported optimism, and no amount of tuning reconstructs the missing hidden path.

Visualization:

![Linear intrinsic beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/beta_sweep_values.png)

![Linear intrinsic beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

### 4. Linear Near-Intrinsic Rare-Support Task

Artifacts:

- [resolved config](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/resolved_config.json)
- [summary](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/summary.json)
- [metrics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/metrics.csv)
- [value plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/beta_sweep_values.png)
- [diagnostic plot](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

Best beta by dataset size:

- `10`: best `beta = 1.2`, value `0.800`, improvement over greedy `+0.040`
- `20`: best `beta = 0.8`, value `0.800`, improvement `+0.064`
- `50`: best `beta = 1.2`, value `0.800`, improvement `+0.056`
- `100`: best `beta = 0.8`, value `0.800`, improvement `+0.004`
- `200`: best `beta = 0.0`, value `0.828`, improvement `+0.000`
- `500`: best `beta = 0.0`, value `0.894`, improvement `+0.000`

Interpretation:

- This is the clean linear turnover we wanted.
- Positive `beta` is best while evidence is still sparse, but by `200` episodes the rare hidden continuation has enough support that extra pessimism becomes strictly harmful.
- The high-coverage optimum is not a pessimistic plateau but recovery by `beta=0`, which matches the rare-support story from the tabular phase.

Visualization:

![Linear near-intrinsic beta sweep values](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/beta_sweep_values.png)

![Linear near-intrinsic beta sweep diagnostics](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/results/raw/linear_near_intrinsic_beta_sweep/beta_sweep_diagnostics.png)

## Interpretation
The beta sweeps sharpen the story from the fixed-beta report:

- In the linear bandit, pessimism helps, but only with careful tuning, and the preferred `beta` changes with dataset size.
- In the linear branching task, the best value is essentially a flat safe-policy plateau once `beta` is high enough, which is exactly what we would expect from a support-limited problem.
- In `linear_intrinsic`, the same support-limited story still holds, but function approximation introduces a small amount of extrapolation leakage at high coverage.
- In `linear_near_intrinsic`, the preferred `beta` flips to zero once rare evidence accumulates, which is the strongest sign yet that the linear suite now captures both graceful failure and over-conservatism.
- Compared with the tabular phase, the linear phase is already more realistic and more nuanced. That is a good sign for the later EHR extension, where feature design and regularization will matter at least as much as the offline support pattern itself.

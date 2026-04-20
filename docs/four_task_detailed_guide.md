# Four-Task Detailed Guide

This guide is the missing bridge between the high-level reports and the code/configs.

Use it when you want to answer any of these questions quickly:

- what exactly are the four core tasks?
- which dataset/config powers each task?
- what is each task meant to prove?
- what code path runs each experiment?
- what should I read first for tabular versus linear?

## Short Answer

If you only want the results story:

- tabular: read `docs/phase1_tabular_report.md`, then `docs/beta_sweep_report.md`
- linear: read `docs/phase2_linear_report.md`, then `docs/linear_beta_sweep_report.md`
- cross-suite synthesis: read `docs/cross_suite_analysis_report.md`

If you also want to understand engineering details, do one more pass through:

- `docs/tabular_quickstart.md` or `docs/linear_quickstart.md`
- the relevant `configs/*.json`
- the task builders in `src/peorl/envs/tabular.py` or `src/peorl/linear/envs.py`
- the experiment runners in `scripts/run_tabular_experiment.py`, `scripts/run_beta_sweep.py`, `scripts/run_linear_experiment.py`, and `scripts/run_linear_beta_sweep.py`

So the answer to "anything left?" is:

- for interpretation only: not much beyond report then sweep
- for implementation or extension work: yes, read the quickstart, configs, and task builders too

## Recommended Reading Order

### Tabular

1. `docs/tabular_quickstart.md`
2. `docs/phase1_tabular_report.md`
3. `docs/beta_sweep_report.md`
4. `configs/bandit_study.json`, `configs/branching_study.json`, `configs/intrinsic_study.json`, `configs/near_intrinsic_study.json`
5. `src/peorl/envs/tabular.py`
6. `src/peorl/experiments.py`

### Linear

1. `docs/linear_quickstart.md`
2. `docs/phase2_linear_report.md`
3. `docs/linear_beta_sweep_report.md`
4. `configs/linear_bandit_study.json`, `configs/linear_branching_study.json`, `configs/linear_intrinsic_study.json`, `configs/linear_near_intrinsic_study.json`
5. `src/peorl/linear/envs.py`
6. `src/peorl/linear/experiments.py`

## Common Experimental Pipeline

The project uses the same high-level pipeline for both tabular and linear experiments.

1. Build a synthetic task with known rewards, transitions, and a fixed behavior policy.
2. Collect an offline dataset by rolling out the behavior policy for a chosen number of episodes.
3. Fit three planners on that logged dataset:
   - `greedy`
   - `pessimistic`
   - `support_masked`
4. Evaluate each learned policy exactly on the known simulator.
5. Save metrics, summary tables, resolved config, and plots into `results/raw/...`.
6. Aggregate fixed-beta results into the main report and beta-sweep results into the sweep report.

### Tabular code path

- task builder: `src/peorl/envs/tabular.py`
- dataset + empirical model: `src/peorl/datasets.py`
- planners: `src/peorl/algorithms/tabular_vi.py`
- diagnostics + exact evaluation: `src/peorl/evaluation.py`
- per-seed experiment loop: `src/peorl/experiments.py`
- report-generating runners:
  - `scripts/run_tabular_experiment.py`
  - `scripts/run_beta_sweep.py`

### Linear code path

- task builder: `src/peorl/linear/envs.py`
- dataset collection: `src/peorl/linear/data.py`
- regression + confidence widths: `src/peorl/linear/regression.py`
- planners: `src/peorl/linear/algorithms.py`
- diagnostics + exact evaluation: `src/peorl/linear/evaluation.py`
- per-seed experiment loop: `src/peorl/linear/experiments.py`
- report-generating runners:
  - `scripts/run_linear_experiment.py`
  - `scripts/run_linear_beta_sweep.py`

## Common Dataset Settings

These settings are shared across almost all fixed studies and sweeps.

### Tabular

- dataset sizes: `10, 20, 50, 100, 200, 500`
- seeds: `200`
- fixed-study pessimism: `beta = 0.8`
- sweep values: `beta in {0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0}`

### Linear

- dataset sizes: `10, 20, 50, 100, 200, 500`
- seeds: `200`
- ridge: `1.0`
- fixed-study pessimism: `beta = 0.8`
- sweep values: `beta in {0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0}`

## The Four Core Tasks

The four-task story is easiest to understand if you treat the tabular suite as the clean conceptual version and the linear suite as the feature-based extension of the same boundary cases.

### 1. Bandit

#### Goal of the task

Show the smallest possible spurious-correlation failure:

- the logged dataset rarely samples one distractor action
- that distractor can look lucky under small data
- greedy may chase it
- pessimism should suppress that mistake

#### Tabular dataset and config

- config: `configs/bandit_study.json`
- sweep config: `configs/bandit_beta_sweep.json`
- task builder: `make_bandit_task(...)` in `src/peorl/envs/tabular.py`
- key task parameters:
  - `num_actions = 8`
  - `best_reward = 0.78`
  - `distractor_reward = 0.62`
  - `distractor_behavior_prob = 0.05`
  - `support_threshold = 3`

#### Linear dataset and config

- config: `configs/linear_bandit_study.json`
- sweep config: `configs/linear_bandit_beta_sweep.json`
- task builder: `make_linear_bandit_task(...)` in `src/peorl/linear/envs.py`
- key task parameters:
  - `feature_dim = 6`
  - `support_threshold = 3`
  - same reward pattern as the tabular bandit, but now represented through shared features

#### Engineering pipeline

1. Build the one-step bandit task.
2. Roll out the fixed behavior policy to create offline data.
3. Fit greedy, pessimistic, and support-masked planners.
4. Evaluate exact return and low-support action mass.
5. In the sweep, vary `beta` to see how much conservatism is needed to suppress the weakly covered distractor.

#### What this task is for

- best entry point for the spurious-correlation story
- easiest task to debug
- easiest task to use when checking whether bonuses or support masks are wired correctly

### 2. Branching

#### Goal of the task

Show an end-to-end planning effect, not just one-step action filtering.

In the tabular suite, this is the clean "safe covered branch versus risky weakly covered branch" planning task.

In the linear suite, the current `linear_branching` task is slightly different in flavor:

- it already behaves more like a support-limited branching family task
- once pessimism is high enough, it settles on the supported safe branch plateau

So tabular `branching` and linear `linear_branching` are related, but they are not perfectly symmetric experiments.

#### Tabular dataset and config

- config: `configs/branching_study.json`
- sweep config: `configs/branching_beta_sweep.json`
- task builder: `make_branching_task(...)` in `src/peorl/envs/tabular.py`
- key task parameters:
  - `root_risky_prob = 0.08`
  - `risky_success_reward = 0.45`
  - `support_threshold = 2`

#### Linear dataset and config

- config: `configs/linear_branching_study.json`
- sweep config: `configs/linear_branching_beta_sweep.json`
- task builder: `make_linear_branching_task(...)` in `src/peorl/linear/envs.py`
- key task parameters:
  - `feature_dim = 6`
  - `support_threshold = 2`
- important default behavior from the builder:
  - `root_risky_prob = 0.10`
  - `hidden_optimal_behavior_prob = 0.10`

#### Engineering pipeline

1. Build a horizon-3 branching MDP.
2. Collect offline trajectories from a behavior policy that prefers the safe side.
3. Estimate values backward through the horizon.
4. Compare whether the planner stays on the safe supported branch or jumps to the risky branch.
5. In the sweep, inspect how much pessimism is needed before the policy stops taking the risky root action.

#### What this task is for

- tabular: cleanest demonstration that pessimism improves multi-step planning
- linear: bridge from one-step bandit to feature-based branching behavior

### 3. Intrinsic

#### Goal of the task

Show graceful failure under genuinely missing support.

This is the task where the optimal branch exists in the true environment but is absent from the logged data. The right conclusion is not "pessimism finds the optimum anyway." The right conclusion is:

- greedy overreaches
- pessimism protects against unsupported optimism
- neither method can truly recover what the dataset never showed

#### Tabular dataset and config

- config: `configs/intrinsic_study.json`
- sweep config: `configs/intrinsic_beta_sweep.json`
- task builder: `make_intrinsic_uncertainty_task(...)` in `src/peorl/envs/tabular.py`
- key task parameters:
  - `root_risky_prob = 0.06`
  - `hidden_optimal_behavior_prob = 0.0`
  - `support_threshold = 2`

#### Linear dataset and config

- config: `configs/linear_intrinsic_study.json`
- sweep config: `configs/linear_intrinsic_beta_sweep.json`
- task builder: `make_linear_intrinsic_task(...)` in `src/peorl/linear/envs.py`
- key task parameters:
  - `feature_dim = 8`
  - `root_risky_prob = 0.04`
  - `hidden_optimal_behavior_prob = 0.0`
  - `support_threshold = 2`

#### Engineering pipeline

1. Build a horizon-3 task with a hidden optimal continuation.
2. Force the behavior policy to never show that continuation.
3. Fit the three planners on the logged data only.
4. Evaluate whether the learned policies plateau at the best supported safe value instead of the true hidden optimum.
5. Use the sweep to verify that tuning `beta` changes robustness but does not remove the support barrier.

#### What this task is for

- this is the key "limit of pessimism" task
- strongest task for separating spurious correlation from true missing-support error

### 4. Near-Intrinsic

#### Goal of the task

Show the turnover point where rare support becomes recoverable and fixed pessimism can become too conservative.

This task differs from `intrinsic` in only one essential way:

- the hidden optimal continuation is now seen rarely instead of never

That makes it the most important task for studying when pessimism should eventually back off.

#### Tabular dataset and config

- config: `configs/near_intrinsic_study.json`
- sweep config: `configs/near_intrinsic_beta_sweep.json`
- task builder: `make_near_intrinsic_task(...)` in `src/peorl/envs/tabular.py`
- key task parameters:
  - `root_risky_prob = 0.10`
  - `hidden_optimal_behavior_prob = 0.10`
  - `support_threshold = 2`

#### Linear dataset and config

- config: `configs/linear_near_intrinsic_study.json`
- sweep config: `configs/linear_near_intrinsic_beta_sweep.json`
- task builder: `make_linear_near_intrinsic_task(...)` in `src/peorl/linear/envs.py`
- key task parameters:
  - `feature_dim = 8`
  - `root_risky_prob = 0.10`
  - `hidden_optimal_behavior_prob = 0.10`
  - `support_threshold = 2`

#### Engineering pipeline

1. Start from the intrinsic task.
2. Change the behavior policy so the hidden continuation is rare rather than absent.
3. Re-run fixed-beta studies to see how a single `beta` behaves across dataset sizes.
4. Run the sweep to locate the regime change where `beta > 0` stops helping and `beta = 0` becomes best.

#### What this task is for

- best task for understanding over-conservatism
- best task for motivating adaptive or data-dependent pessimism
- most useful precursor to any realistic dataset where support is sparse but not literally zero

## How To Read The Reports Task By Task

For each task, the fixed report and sweep answer different questions.

### Fixed report

Use the fixed report to answer:

- what happens at the default `beta = 0.8`?
- is pessimism better than greedy at low coverage?
- does support-masking behave similarly or differently?
- what value plateau does the task settle into?

Relevant files:

- tabular fixed report: `docs/phase1_tabular_report.md`
- linear fixed report: `docs/phase2_linear_report.md`

### Sweep report

Use the sweep report to answer:

- how sensitive is the task to `beta`?
- is the best `beta` stable or dataset-size dependent?
- does stronger pessimism keep helping, saturate, or become harmful?

Relevant files:

- tabular sweep: `docs/beta_sweep_report.md`
- linear sweep: `docs/linear_beta_sweep_report.md`

## What Is Left Beyond The Reports?

If your goal is just to summarize the project, the reports are already enough.

If your goal is to modify or extend the project, there are still four things worth checking after the reports:

1. Quickstarts for the big-picture map:
   - `docs/tabular_quickstart.md`
   - `docs/linear_quickstart.md`
2. Configs for the exact dataset knobs:
   - all `configs/*study.json`
   - all `configs/*beta_sweep.json`
3. Task builders for the true environment definition:
   - `src/peorl/envs/tabular.py`
   - `src/peorl/linear/envs.py`
4. Saved artifacts for one concrete run:
   - `results/raw/<task>/resolved_config.json`
   - `results/raw/<task>/summary.json`
   - `results/raw/<task>/metrics.csv`

That last step matters because some details are clearer in the raw artifacts than in the prose reports.

## Practical Recommendation

If you are organizing your own reading or presentation notes, use this minimal checklist:

### For tabular

1. Read `docs/phase1_tabular_report.md`
2. Read `docs/beta_sweep_report.md`
3. Skim `configs/*.json`
4. Open `src/peorl/envs/tabular.py` only if you need exact task mechanics

### For linear

1. Read `docs/phase2_linear_report.md`
2. Read `docs/linear_beta_sweep_report.md`
3. Skim `configs/linear_*.json`
4. Open `src/peorl/linear/envs.py` only if you need exact feature/task mechanics

That is the shortest path that still preserves both the scientific story and the implementation story.

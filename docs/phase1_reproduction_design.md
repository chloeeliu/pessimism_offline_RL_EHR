# Phase 1 Design: Reproducing PEVI in a Minimal Offline RL Setting

## Goal

This project starts from the paper *Is Pessimism Provably Efficient for Offline RL?* (Jin, Yang, Wang; ICML 2021 / MOR 2024) and focuses first on a clean, minimal reproduction of the paper's core idea:

- a naive greedy offline value-learning baseline can fail under limited dataset coverage because it overfits to noisy or weakly supported state-action values
- a pessimistic value iteration method can suppress this failure mode by subtracting an uncertainty penalty

The first milestone is not a large-scale benchmark. It is a controlled reproduction that makes the paper's mechanism visible and measurable.

## What We Want to Validate First

The paper argues that offline RL suffers from two different effects under weak coverage:

- **intrinsic uncertainty**: the dataset may miss trajectories that matter for the optimal policy
- **spurious correlation**: the learner may prefer actions that look good only because of estimation error on weakly covered parts of the dataset

The key claim we want to reproduce in code is:

- **pessimism reduces the damage from spurious correlation**
- **the remaining error should mainly track true lack of coverage, not accidental overestimation**

For Phase 1, the project should make that claim visible in simple environments before we move to more realistic offline RL datasets or EHR data.

## Recommended Reproduction Scope

### Stage A: Smallest possible demonstration

Start with a toy setting where failure is easy to see and cheap to rerun.

Recommended environments:

- **multi-armed bandit**
  - useful as the smallest case of spurious correlation
  - easy to visualize: weakly sampled actions can look falsely optimal
- **small tabular episodic MDP**
  - horizon 3 to 10
  - deterministic or low-stochastic transitions
  - hand-designed bottlenecks / branches so coverage can be controlled

Why start here:

- ground-truth optimal value is known exactly
- we can vary coverage systematically
- runs are fast enough for many seeds
- plots will be interpretable

### Stage B: Theory-faithful offline RL setting

Implement a tabular or linear-feature version of the paper's pessimistic value iteration recipe.

Recommended progression:

1. **tabular PEVI-like plug-in estimator**
   - empirical Bellman backup
   - uncertainty penalty based on counts
   - greedy policy extraction from penalized Q
2. **linear MDP approximation**
   - shared features for state-action pairs
   - ridge-style least-squares value backup
   - uncertainty penalty based on feature covariance

The tabular version is the fastest path to a convincing reproduction. The linear version is the closest bridge to the paper's formal setup.

### Stage C: Optional benchmark sanity check

After the synthetic reproduction works, add one benchmark-style offline dataset as a sanity check, not as the main proof point.

Recommended option:

- **Minari / D4RL-derived simple navigation datasets**, especially small discrete or low-dimensional tasks

Reason:

- easier operationally than jumping straight into MuJoCo locomotion
- still lets us test the code on a standardized offline dataset interface
- preserves focus on the pessimism mechanism rather than deep-RL engineering

I would avoid starting with large continuous-control benchmarks in the first pass. They will make debugging much harder and can blur whether failures come from the idea or from function approximation and optimization details.

## Proposed Algorithms

### Primary method

- **PEVI-style pessimistic value iteration**
  - estimate Q by backward dynamic programming
  - subtract an uncertainty penalty from the Bellman target or estimated Q
  - act greedily with respect to the penalized estimate

For Phase 1, the implementation should stay intentionally simple:

- tabular counts for uncertainty in tabular MDPs
- linear confidence widths for linear-feature MDPs
- no deep networks yet

### Comparison baselines

At minimum:

- **greedy plug-in baseline**
  - same estimator without pessimistic penalty
  - this is the most important comparator because it isolates the effect of pessimism

Useful optional baselines:

- **behavior policy**
  - important sanity baseline on offline data
- **behavior cloning**
  - especially helpful if the dataset is strongly policy-driven
- **uniformly conservative baseline**
  - e.g. fixed penalty, clipped values, or action masking by support threshold
  - useful to show that targeted uncertainty penalties matter more than generic conservatism

## Datasets and Data Generation

### Curated synthetic offline datasets

This should be the main data source for Phase 1.

Construct datasets by rolling out fixed behavior policies with controllable coverage:

- **well-covered dataset**
  - broad action support
  - should reduce the gap between greedy and pessimistic methods
- **biased / weak-coverage dataset**
  - low support on some actions or trajectory branches
  - should expose greedy overestimation
- **mixed-quality dataset**
  - combination of suboptimal and partially competent behavior policies
  - useful to mimic more realistic offline data

Key knobs:

- number of episodes
- behavior policy entropy
- whether rare actions are optimal, suboptimal, or distractors
- horizon
- reward noise
- transition stochasticity

### Existing offline RL dataset option

Use this only after the curated setup works.

Recommended direction:

- **Minari**, which provides a current offline RL dataset interface for Gymnasium-style environments
- optionally use **D4RL-derived datasets exposed through Minari** for standardized comparisons

This should be treated as a portability check for our pipeline, not the main reproduction target.

## Success Criteria

The first reproduction is successful if we can show the following pattern consistently across seeds:

1. Under weak coverage, the greedy plug-in baseline overestimates value and selects poorly supported actions too often.
2. The pessimistic method chooses safer, better-supported actions and improves true return.
3. As coverage improves, the gap between greedy and pessimistic methods shrinks.
4. In settings where the optimal trajectory is genuinely missing from the dataset, pessimism does not magically recover optimal performance; it fails gracefully.

That last point matters. A good reproduction should show both the strength and the limit of pessimism.

## Metrics

### Primary metrics

- **true policy value** under the simulator
- **suboptimality gap** relative to the optimal policy
- **policy agreement with the optimal policy**
- **frequency of selecting low-support actions**

### Diagnostic metrics

- estimated Q versus true Q
- uncertainty penalty magnitude by step / state-action pair
- visitation counts or feature coverage along the chosen trajectory
- calibration plot: weak support versus overestimation error

### Visualization ideas

- value versus dataset size
- value versus coverage level
- histogram of chosen actions under greedy vs pessimistic
- trajectory heatmaps for visited states and selected actions
- scatter plot of support count versus estimation error

## High-Level Experimental Design

### Experiment family 1: Spurious correlation in the smallest setting

Environment:

- multi-armed bandit with one good arm, several mediocre arms, and one weakly sampled distractor

Question:

- does greedy selection over-pick the weakly sampled distractor?

Expected outcome:

- yes for the greedy estimator
- much less so for pessimistic selection

### Experiment family 2: Offline planning in a small episodic MDP

Environment:

- branching tabular MDP with a rewarding path that requires correct early decisions

Question:

- does pessimism improve end-to-end offline planning when some branches are poorly covered?

Expected outcome:

- pessimistic value iteration avoids unsupported optimistic branches
- performance gain is largest in low-coverage regimes

### Experiment family 3: Coverage sweep

Environment:

- same MDP, same reward structure

Intervention:

- vary dataset size and behavior policy entropy

Question:

- how does the greedy-vs-pessimistic gap evolve as support improves?

Expected outcome:

- gap is largest when support is weak
- gap narrows when support becomes broad

### Experiment family 4: Linear-feature extension

Environment:

- synthetic linear MDP or feature-based finite-horizon environment

Question:

- does a linear-confidence version of pessimism preserve the same qualitative behavior?

Expected outcome:

- yes, though sensitivity to regularization and feature quality will be higher

## Initial Repository Design

Suggested structure:

```text
pessimism_offline_RL/
  docs/
    phase1_reproduction_design.md
    experiment_notes.md
  src/
    peorl/
      envs/
      datasets/
      algorithms/
      evaluation/
      plotting/
      utils/
  configs/
    env/
    data/
    algo/
    experiment/
  scripts/
    generate_dataset.py
    run_experiment.py
    summarize_results.py
  notebooks/
    exploratory/
  results/
    raw/
    figures/
```

### Module responsibilities

- `envs/`
  - toy bandit, tabular MDP, optional linear MDP simulator
- `datasets/`
  - offline rollout generation, trajectory serialization, support statistics
- `algorithms/`
  - greedy plug-in baseline
  - pessimistic value iteration
  - optional BC / behavior baseline wrappers
- `evaluation/`
  - exact policy evaluation for small environments
  - Monte Carlo rollout evaluation
  - suboptimality and support diagnostics
- `plotting/`
  - publication-style figures for coverage, return, and overestimation

## Implementation Principles

- prefer **NumPy-first** implementations for the first pass
- keep environments small enough for exact evaluation
- make coverage explicit and inspectable
- separate dataset generation from algorithm training
- make all experiments config-driven and seed-controlled

For the first pass, the code should optimize for interpretability, not throughput.

## Concrete Phase Plan

### Phase 0: Paper-to-code alignment

- write down the exact PEVI update we will implement in tabular form
- define the greedy baseline as the same update without the penalty
- define a small set of controlled environments

Deliverable:

- short implementation note with equations and notation mapping

### Phase 1: Tabular synthetic reproduction

- implement bandit and small episodic MDP
- generate offline datasets with coverage control
- run greedy vs pessimistic comparison
- produce first plots

Deliverable:

- minimal reproducible experiment suite with figures

### Phase 2: Linear-feature extension

- implement feature-based MDP
- add linear least-squares backup and uncertainty penalty
- stress-test sensitivity to feature quality and regularization

Deliverable:

- second experiment suite closer to the paper's formal setting

### Phase 3: Benchmark portability check

- add Minari-backed dataset loading
- run one lightweight benchmark environment
- verify pipeline works outside fully synthetic settings

Deliverable:

- one benchmark experiment and one comparison figure

### Phase 4: Bridge to sepsis rolling monitoring

Not the focus now, but the bridge should look like this:

- replace simulator rollouts with logged patient trajectories
- define finite-horizon monitoring / intervention windows
- reinterpret uncertainty penalty as protection against unsupported treatment-state decisions
- use the synthetic phase to decide which diagnostics are essential before touching EHR data

## Main Risks

### Risk 1: Jumping too early into deep offline RL

If we start with MuJoCo-scale function approximation, we may not be able to tell whether a result comes from pessimism, optimizer instability, network architecture, or dataset preprocessing.

Mitigation:

- earn the mechanism first in tabular / linear settings

### Risk 2: Using only average return

Average return alone can hide the actual failure mode.

Mitigation:

- always log support, overestimation, action selection, and coverage diagnostics

### Risk 3: Confusing pessimism with generic conservatism

A weak implementation can look "safe" by simply shrinking all values.

Mitigation:

- compare against both greedy and simple conservative ablations
- inspect whether the penalty tracks uncertainty rather than acting as a blunt constant

## Decisions We Do Not Need From You Yet

Nothing is blocked right now. I can proceed with a lightweight first implementation under the following assumptions:

- Python + NumPy-first stack
- synthetic environments before benchmark datasets
- tabular reproduction before linear-feature extension

## Decisions That Will Matter Soon

These are worth confirming once we move from planning to code:

- whether you want the first implementation to stay fully tabular or include the linear-feature version immediately
- whether to use a simple script/config stack or bring in Hydra from day one
- whether the first benchmark portability check should use Minari only or D4RL-compatible tasks as well

## Recommended Next Step

Build the smallest experiment ladder in this order:

1. bandit counterexample
2. branching tabular MDP
3. coverage sweep
4. linear-feature extension

That gives us a clean story, fast iteration, and a strong foundation before we touch sepsis data.

## Notes and Sources

- Paper page: https://proceedings.mlr.press/v139/jin21e.html
- Paper PDF: https://proceedings.mlr.press/v139/jin21e/jin21e.pdf
- Minari docs: https://minari.farama.org/


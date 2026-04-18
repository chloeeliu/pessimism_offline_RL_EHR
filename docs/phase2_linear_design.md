# Phase 2 Design: Linear MDP Experiments

## Goal

Phase 2 moves the project from tabular diagnostic tasks toward the linear-feature setting that is much closer to the paper's formal assumptions.

The purpose of this phase is not to jump straight to complex function approximation. It is to answer a narrower question:

- do the main tabular conclusions survive when values must be inferred through shared features rather than explicit counts?

This phase should preserve the main strengths of the tabular work:

- interpretable synthetic tasks
- exact or near-exact evaluation
- explicit coverage control
- diagnostics that separate spurious correlation from genuine missing support

## Why Move Now

The current tabular suite already covers the key qualitative cases:

- spurious-correlation failure
- clean planning benefit
- unrecoverable off-support failure
- near-missing-support transition where fixed pessimism eventually hurts

More tabular work would refine the boundary between these regimes, but it is unlikely to change the high-level story. A linear-feature phase is the next useful step because it:

- aligns better with the paper
- tests whether shared representation changes the coverage story
- provides a more realistic bridge toward EHR-style state spaces

## Target Setting

We should begin with finite-horizon synthetic linear MDPs where the feature map is known and controlled.

Core assumptions for the first implementation:

- finite horizon `H`
- small discrete action set
- state-action feature vector `phi(s, a) in R^d`
- linear reward model
- linear next-value expectation model
- ridge-regularized least-squares fitting at each step

This keeps the implementation mathematically close to the paper while remaining simple enough to debug.

## Recommended Experiment Ladder

### Stage A: Linear bandit sanity check

Start with the one-step case:

- contextual or feature-based bandit
- one weakly covered distractor region in feature space
- same greedy versus pessimistic comparison as the tabular bandit

Purpose:

- verify feature covariance and confidence bonuses are wired correctly

### Stage B: Linear episodic planning task

Create a finite-horizon synthetic MDP with:

- a safe branch
- a risky branch
- shared features between supported and weakly supported actions

Purpose:

- test whether pessimism still prevents over-optimistic planning when the learner must generalize through features

### Stage C: Off-support and near-support linear tasks

Lift the tabular intrinsic and near-intrinsic logic into the linear setting:

- one regime where the optimal continuation is effectively missing
- one regime where it is present very rarely

Purpose:

- test whether the same graceful-failure and over-conservatism patterns survive under shared representation

## Proposed Algorithms

### Primary method

- linear PEVI-style pessimistic value iteration

At each step:

1. fit a ridge-regularized linear model for the Bellman target
2. estimate `Q_hat_h(s, a) = <phi(s, a), theta_h>`
3. subtract a confidence width based on the inverse feature covariance
4. act greedily with respect to the penalized estimate

### Baselines

- linear greedy plug-in estimator
- behavior policy
- optional support-masked or clipped-confidence conservative baseline

The greedy baseline should reuse the exact same regression pipeline without the pessimistic penalty.

## Metrics

Retain the tabular metrics where possible:

- true policy value
- suboptimality gap
- agreement with the optimal policy
- occupancy mass on weak-support regions
- chosen-action overestimation
- chosen-action pessimistic bonus

Add linear-specific diagnostics:

- effective feature coverage, e.g. `phi^T Lambda^-1 phi`
- feature-space novelty of chosen actions
- condition number or trace diagnostics for stepwise covariance matrices

## Environment Design Principles

### Keep exact evaluation available

Even if planning uses features, the underlying simulator should stay small enough for exact dynamic programming.

### Separate feature sharing from transition complexity

The first linear environments should not be hard because of complicated dynamics. They should be hard because:

- weak coverage propagates through shared features
- the learner must extrapolate in feature space

### Make coverage knobs explicit

Every linear environment should expose:

- feature dimension
- behavior policy support level
- hidden-support probability for rare branches
- reward noise level
- transition stochasticity

## Initial Code Plan

Add a new linear subpackage that mirrors the existing tabular organization:

```text
src/peorl/linear/
  __init__.py
  types.py
  envs.py
  data.py
  regression.py
  algorithms.py
  evaluation.py
```

Responsibilities:

- `types.py`
  - dataclasses for linear tasks, transitions, datasets, and planner outputs
- `envs.py`
  - synthetic linear bandit and linear branching builders
- `data.py`
  - dataset collection and Bellman-target assembly
- `regression.py`
  - ridge solve, covariance inverse, confidence width utilities
- `algorithms.py`
  - greedy and pessimistic linear PEVI interfaces
- `evaluation.py`
  - exact evaluation against the known simulator plus feature-coverage diagnostics

## Milestones

### Milestone 1: Scaffold and feature plumbing

- define linear task dataclasses
- create one synthetic linear bandit and one linear branching builder
- add ridge-regression helpers and confidence-width utilities

Deliverable:

- code scaffold with working dataset generation and regression utilities

### Milestone 2: First working linear planner

- implement one-step linear greedy and pessimistic planners
- run a linear bandit experiment

Deliverable:

- first end-to-end linear result with plots

### Milestone 3: Finite-horizon linear PEVI

- implement backward dynamic programming with linear Bellman targets
- run synthetic linear branching experiments

Deliverable:

- fixed-beta and beta-sweep reports for linear tasks

### Milestone 4: Boundary study

- add off-support and near-support linear families
- compare where fixed pessimism helps, hurts, or saturates

Deliverable:

- linear analogue of the tabular phase diagram story

## Immediate Recommendation

The next implementation step should be:

1. build the linear scaffold
2. implement linear bandit first
3. only then move to finite-horizon linear PEVI

That keeps the jump from tabular to linear controlled and testable.

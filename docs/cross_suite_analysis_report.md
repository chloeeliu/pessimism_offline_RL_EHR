# Cross-Suite Analysis Report

This report synthesizes the current experimental evidence across:

- [phase1_tabular_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/phase1_tabular_report.md)
- [beta_sweep_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/beta_sweep_report.md)
- [phase2_linear_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/phase2_linear_report.md)
- [linear_beta_sweep_report.md](/Users/chloe/Desktop/uw_madison/26Spring/RL/pessimism_offline_RL/docs/linear_beta_sweep_report.md)

The goal is not to restate every result, but to answer the higher-level questions that only become clear when the four reports are read together:

- which conclusions are stable across tabular and linear settings?
- where does the linear phase materially change the story?
- what does the sweep evidence say about fixed `beta = 0.8`?
- which tasks are really demonstrating spurious correlation, support limits, or over-conservatism?
- what is the most justified next step for the project?

## Executive Summary

The current evidence supports five strong conclusions.

1. Pessimism clearly helps under weak support, but the reason differs by task.
   - In bandit-like settings, it suppresses spurious correlation.
   - In branching settings, it stabilizes multi-step planning.
   - In intrinsic tasks, it protects the best supported value without recovering missing support.

2. The tabular suite is cleaner and more interpretable than the linear suite, but the core mechanism survives the move to shared features.
   - Greedy still overuses weakly supported actions.
   - Pessimistic methods still reduce low-support mass and often improve value.
   - The linear suite introduces stronger sensitivity to tuning and representation.

3. Fixed `beta = 0.8` is a reasonable default, not a universal optimum.
   - It is often strong in low-data regimes.
   - It is rarely the best setting everywhere.
   - It can become too conservative once rare but real evidence accumulates.

4. The intrinsic and near-intrinsic tasks are the most scientifically informative part of the current repository.
   - `intrinsic` shows graceful failure under missing support.
   - `near_intrinsic` shows the turnover where fixed pessimism helps early and hurts later.
   - The same boundary logic appears in both tabular and linear phases.

5. The most justified next step is no longer “more of the same fixed-beta studies.”
   - The project now has enough evidence to motivate adaptive pessimism, hidden-support sweeps, or a move toward EHR-like feature spaces.

## What The Tabular Phase Established

The tabular phase is still the cleanest proof-of-mechanism layer in the repo.

### Stable tabular conclusions

- In `bandit`, greedy overestimates weakly covered distractors and pessimism reliably improves value.
- In `branching`, pessimism improves true multi-step planning rather than just one-step action filtering.
- In `intrinsic`, pessimism reaches the best supported policy value `0.800` but cannot recover the hidden optimum `0.950`.
- In `near_intrinsic`, fixed pessimism helps at tiny data but becomes the wrong inductive bias once rare support accumulates.

### Why the tabular phase matters

The tabular tasks are small enough that the causal story is visible:

- support is explicit
- evaluation is exact
- uncertainty penalties are inspectable
- performance shifts can be traced to concrete coverage patterns instead of representation error

That makes the tabular phase the strongest place to defend the main claim:

- pessimism is most useful when the learner would otherwise over-trust weak support
- pessimism does not solve genuine missing-support limits

## What The Linear Phase Changed

The linear phase keeps the same qualitative agenda, but it introduces a more realistic failure mode: representation changes the effective strength of the same nominal penalty.

### Stable linear conclusions

- Weak support still causes problems under feature sharing.
- Pessimism still suppresses low-support behavior.
- The safe-supported plateau still appears in off-support tasks.
- The rare-support turnover still appears once hidden evidence accumulates.

### New linear nuances

- The linear bandit is much more beta-sensitive than the tabular bandit.
- The linear intrinsic task shows mild extrapolation leakage above the safe plateau at high data, even though the hidden optimum is still not actually recovered.
- The current `linear_branching` task behaves more like a support-limited safe-branch plateau than like the clean tabular planning demo.
- Representation and covariance structure matter enough that “stronger pessimism” is no longer close to monotone improvement.

This is important because it means the project has already crossed a conceptual boundary:

- in tabular, pessimism is mostly about support counts
- in linear, pessimism is about support filtered through feature geometry

That is much closer to the setting we would eventually face in EHR-style state representations.

## Task-By-Task Cross-Suite Reading

## 1. Bandit Family

### What the combined evidence says

The bandit family is still the cleanest spurious-correlation test, but the linear version is less forgiving.

### Tabular bandit

- Fixed `beta = 0.8` already works well.
- The sweep suggests strong pessimism is consistently helpful.
- Best values are reached with relatively high `beta`, often near the top of the tested range.

This means the tabular bandit behaves like a pure weak-support distractor problem:

- more penalty mostly means less overestimation
- less overestimation mostly means better decisions

### Linear bandit

- Fixed `beta = 0.8` is too conservative at `10` episodes.
- The sweep is non-monotone.
- `beta = 2.0` is best at `10`, but much too strong by `20`.
- `beta = 0.8` becomes useful in the middle regime, then unnecessary by `500`.

### Combined interpretation

This is the clearest sign that the linear phase is not just a noisier copy of tabular.

In the tabular bandit:

- support counts map directly to the needed level of conservatism

In the linear bandit:

- feature sharing alters the effective penalty
- the same numeric `beta` can be too weak in one regime and too strong in another

So the bandit family gives a crisp overall lesson:

- the spurious-correlation mechanism survives the move to linear features
- but beta calibration becomes meaningfully harder under function approximation

## 2. Branching Family

### What the combined evidence says

The branching family splits into two slightly different stories.

### Tabular branching

- This is the cleanest end-to-end planning success case.
- Pessimism keeps the policy on the safe branch early and reaches near-optimal value quickly.
- The sweep shows moderate pessimism is best at low coverage and decays smoothly toward zero as coverage improves.

This is exactly the behavior we would want from a good offline planning regularizer:

- strong enough to block unsupported optimism when data is scarce
- weak enough to disappear once support is abundant

### Linear branching

- Fixed `beta = 0.8` immediately locks onto the safe supported value `0.800`.
- The sweep shows a broad plateau: once `beta` is large enough, changing it barely affects value.
- The task behaves more like a support-limited safe-policy problem than like a genuine recovery-to-optimum planning task.

### Combined interpretation

The branching family tells us two useful things.

First:

- the tabular branching task is the better demonstration of “pessimism improves planning”

Second:

- the current linear branching task is already acting like a support-limit diagnostic

That is not a bad result, but it changes how the task should be described:

- tabular `branching` is a planning win
- linear `linear_branching` is closer to a stable safe-plateau sanity check

## 3. Intrinsic Family

### What the combined evidence says

The intrinsic family is the strongest evidence in the repo that pessimism is useful but not magical.

### Tabular intrinsic

- Best supported value is `0.800`.
- True optimum is `0.950`.
- No tested `beta` exceeds the support ceiling.
- The sweep is comparatively flat because the main limitation is missing support, not poor beta choice.

### Linear intrinsic

- The same safe plateau dominates most of the study.
- The sweep eventually prefers `beta = 0.0` at `500` episodes, but only because the greedy estimator leaks slightly above `0.800`.
- That leakage is not true recovery of the hidden optimum.

### Combined interpretation

This family gives the cleanest answer to a very important question:

- what should pessimism do when the data never shows the right continuation?

The answer is:

- avoid unsupported optimism
- settle on the best clearly supported branch
- do not pretend to reconstruct absent evidence

The linear version adds one nuance:

- under function approximation, the model can leak upward beyond the safe plateau without actually learning the hidden optimal branch

That makes the linear intrinsic task especially useful for future work, because it separates two concepts:

- support failure
- extrapolation or misspecification leakage

## 4. Near-Intrinsic Family

### What the combined evidence says

The near-intrinsic family is the most important task family in the current project.

### Tabular near-intrinsic

- Fixed `beta = 0.8` helps early.
- By `100` episodes, greedy overtakes fixed pessimism.
- The sweep flips to `beta = 0.0` by `100` episodes.

### Linear near-intrinsic

- Fixed `beta = 0.8` again helps early and then gets stuck on the safe plateau.
- The sweep flips to `beta = 0.0` by `200` episodes.
- Greedy and support-masked both recover toward the hidden optimal policy at high coverage.

### Combined interpretation

This family is where the current repository becomes genuinely interesting rather than just confirmatory.

It shows that:

- pessimism is not just a static robustness knob
- its usefulness depends on whether the rare evidence is still effectively noise or has become statistically meaningful

This is the strongest argument in the repo for an adaptive pessimism schedule.

It also gives the most natural bridge to practical settings:

- many realistic offline datasets are not fully off-support
- they are sparse, imbalanced, and weakly informative
- that is exactly the regime where fixed pessimism can help first and hurt later

## What The Sweep Reports Add Beyond The Fixed Reports

The fixed reports answer:

- does `beta = 0.8` work reasonably well?

The sweep reports answer the harder question:

- what type of problem is this task actually representing?

The sweep evidence reveals four distinct beta-response regimes.

### Regime A: Strong-pessimism regime

Representative task:

- tabular bandit

Signature:

- increasing `beta` keeps helping or at least does not meaningfully hurt

Meaning:

- the task is dominated by spurious correlation on weak support

### Regime B: Moderate-then-decay regime

Representative task:

- tabular branching

Signature:

- moderate `beta` is best early
- preferred `beta` shrinks smoothly with more coverage

Meaning:

- pessimism is useful as a temporary planning stabilizer, not a permanent bias

### Regime C: Support-ceiling regime

Representative tasks:

- tabular intrinsic
- much of linear intrinsic
- much of linear branching

Signature:

- many positive betas reach the same supported plateau
- changing `beta` mostly changes conservatism, not value

Meaning:

- the task is limited by missing support, not by under-tuned penalty strength

### Regime D: Turnover regime

Representative tasks:

- tabular near-intrinsic
- linear near-intrinsic

Signature:

- positive `beta` helps early
- `beta = 0.0` becomes best later

Meaning:

- the hidden evidence transitions from noise-like to useful
- fixed pessimism becomes over-conservative

This four-regime view is one of the most useful ways to organize the repo’s current results.

## How Good Is `beta = 0.8` Really?

`beta = 0.8` remains a defensible default, but the combined evidence now lets us say something sharper.

### Why `0.8` is still reasonable

- It is usually strong enough to show the intended pessimism mechanism.
- It performs well in weak-data regimes across many tasks.
- It is often close to the best value in the middle regime.
- It is a good compromise when we want a single setting for a small benchmark suite.

### Why `0.8` is not enough as a research conclusion

- It is too weak for the strongest tabular bandit cases.
- It is too strong for later near-support recovery regimes.
- In linear settings, the same `0.8` can mean very different effective conservatism.

### Best current interpretation

`beta = 0.8` should now be treated as:

- a stable repository default
- not the scientifically preferred final answer

The scientific conclusion is instead:

- beta must depend on task structure, data coverage, and representation class

## Baseline Comparison: What We Learned From `support_masked`

The `support_masked` baseline turned out to be more useful than a throwaway ablation.

### What it proved

- Some gains in the tiny-data regime really do come from simple conservatism.
- A hard threshold can be very competitive when the main problem is obviously low support.

### What it failed to do

- It cannot express graded uncertainty well.
- It can be too blunt in the medium-data regime.
- It does not solve the turnover problem in a principled way.

### Combined interpretation

This strengthens the project’s claims because it shows:

- the advantage of pessimism is not merely “be conservative”
- the advantage is “be conservative in a way that tracks uncertainty structure”

That claim is especially strong in tabular branching and in the sweep evidence.

## Strongest Scientific Claims Supported By Current Evidence

At this point, the repo can defend the following claims with fairly high confidence.

1. Under weak coverage, greedy offline planning over-trusts poorly supported actions.
2. Pessimistic planning reliably reduces low-support action selection.
3. The resulting value gains are strongest when the problem is dominated by spurious correlation or weak-support planning error.
4. Pessimism does not recover trajectories that are fully absent from the dataset.
5. When support is rare but not absent, the best penalty strength changes over time and can eventually drop to zero.
6. Shared features preserve the mechanism but make tuning and interpretation more delicate.

## Weakest Points Or Residual Ambiguities

The current evidence also leaves a few open issues.

### 1. The linear branching task is not the cleanest analogue of tabular branching

It is informative, but it behaves more like a support-limit plateau than like a planning-recovery task.

### 2. Linear extrapolation leakage complicates interpretation

In `linear_intrinsic`, performance slightly above the safe plateau is not true support recovery.

### 3. The study still uses one fixed feature design family

This means we do not yet know how robust the conclusions are to representation misspecification.

### 4. The current beta story is descriptive, not prescriptive

The sweeps show what happens, but they do not yet provide a rule for choosing `beta`.

## Recommended Next Step

The evidence now supports a clear priority order.

### Best next scientific step

Implement an adaptive or data-dependent pessimism rule, then test it first on:

- `near_intrinsic`
- `linear_near_intrinsic`

Those are the tasks where fixed pessimism most clearly fails.

### Best next diagnostic step

Sweep hidden-support probability directly, not just dataset size or `beta`.

That would produce a cleaner phase diagram with axes like:

- dataset size
- hidden-support probability
- best `beta`

### Best next engineering step

Move toward higher-dimensional, EHR-like feature spaces only after preserving the current diagnostics:

- low-support mass
- chosen-action Q error
- agreement with the optimal policy
- feature novelty or covariance-based uncertainty

The project is now at the stage where “harder environments” alone are less valuable than “better controlled transitions between support regimes.”

## Bottom Line

Taken together, the four reports tell a coherent story.

The tabular phase established the mechanism cleanly:

- pessimism helps when weak support creates spurious optimism
- pessimism fails gracefully when support is truly missing
- fixed pessimism eventually becomes too conservative when rare support becomes informative

The linear phase preserved that story while adding the main complication we should expect in realistic settings:

- the right amount of pessimism depends not only on support, but on representation

That means the repository has already outgrown the question “does pessimism help at all?”

The better question now is:

- how should pessimism adapt to support quality and feature geometry so that it protects early without blocking recovery later?

That is the clearest next problem for this codebase to solve.

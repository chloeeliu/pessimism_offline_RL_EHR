# Phase 3 Design: ICU-Only Sepsis EHR Extension on MIMIC-IV

## Goal

Phase 3 extends the pessimism project from synthetic tabular and linear MDPs to a real offline RL setting built from ICU sepsis trajectories in MIMIC-IV.

The main question is:

- does pessimism still help when the learner must recommend monitoring or escalation decisions from logged ICU sepsis trajectories with uneven support?

This phase is meant to extend the pessimism paper in the same spirit as the first two phases:

- keep the action space small and interpretable
- make support mismatch explicit
- compare greedy versus pessimistic policies under the same model class
- look for the same three behaviors:
  - benefit under weak support
  - graceful failure under missing support
  - over-conservatism once rare support accumulates

## Data Source

Primary data source:

- MIMIC-IV DuckDB database:
  - `/Users/chloe/Desktop/healthcare/mimic-iv-3.1/buildmimic/duckdb/mimic4_dk.db`

Relevant schemas confirmed in this build:

- `mimiciv_icu`
- `mimiciv_hosp`
- `mimiciv_derived`

Relevant tables already present:

- ICU stay and timing:
  - `mimiciv_icu.icustays`
- sepsis cohort anchor:
  - `mimiciv_derived.sepsis3`
- organ dysfunction and hourly SOFA:
  - `mimiciv_derived.sofa`
- vitals:
  - `mimiciv_derived.vitalsign`
- blood gas and lactate:
  - `mimiciv_derived.bg`
- chemistry:
  - `mimiciv_derived.chemistry`
- complete blood count:
  - `mimiciv_derived.complete_blood_count`
- ventilation status:
  - `mimiciv_derived.ventilation`
- ICU infusions and fluid input:
  - `mimiciv_icu.inputevents`

Observed cohort scale in this database:

- ICU stays overall: `94,458`
- sepsis-3 positive ICU stays: `41,296`

That is a strong starting point for an ICU-only sepsis study.

## Recommendation in One Line

Start with an ICU-only sepsis rolling monitoring and escalation task over the first 24 hours after sepsis suspicion.

Do not start with direct fluid or vasopressor dosing.

The first EHR action space should answer:

- should this patient remain on the current monitoring path,
- should clinicians intensify review soon,
- or is there evidence of escalation-level instability?

This is the cleanest and safest extension of the pessimism paper.

## How This Extends the Pessimism Paper

The paper’s mechanism is:

- estimate value from offline data
- add a pessimistic penalty tied to uncertainty
- avoid selecting unsupported state-action choices

Our EHR extension should preserve that mechanism exactly, but reinterpret the pieces clinically.

### In the paper

- the learner faces weakly covered state-action regions
- greedy estimation may overreach there
- pessimism should avoid unsupported optimistic choices

### In ICU sepsis

- some clinical states are common and well supported
  - mild hemodynamic instability
  - stable ventilated patients
  - mild inflammation with regular labs
- some states are rare and poorly supported
  - rapid deterioration soon after ICU admission
  - mixed respiratory and circulatory failure with sparse matching precedents
  - unusual combinations of missingness, organ dysfunction, and escalation

So the direct paper-style extension is:

- `greedy`:
  - recommend escalation too aggressively in poorly supported clinical states because fitted values extrapolate
- `pessimistic`:
  - downweight unsupported escalation decisions
- key empirical question:
  - does pessimism help in sparse-support ICU states without becoming too conservative in common states?

## Concrete Clinical Task

### Setting

- ICU-only sepsis episodes
- one decision every `4` hours
- horizon `H = 6`
- total episode length `24` hours after sepsis suspicion time

### Cohort definition

Use:

- `mimiciv_derived.sepsis3`

Keep stays satisfying:

- `sepsis3 = TRUE`
- valid `stay_id`
- valid `suspected_infection_time`
- ICU stay in `mimiciv_icu.icustays`
- enough observed data in the first 24 hours to build rolling windows

Recommended anchor time:

- `t0 = suspected_infection_time`

Recommended episode window:

- `[t0, t0 + 24h)`

Recommended exclusions for the first pass:

- ICU stay shorter than `12` hours after `t0`
- missing all vitals in first `8` hours after `t0`
- extreme data corruption or duplicate-stay conflicts

This gives a clean, clinically interpretable cohort.

## MDP Definition

### Unit of analysis

- one ICU sepsis stay

### Decision times

- every `4` hours after `t0`
- windows:
  - `w0 = [t0, t0+4h)`
  - `w1 = [t0+4h, t0+8h)`
  - ...
  - `w5 = [t0+20h, t0+24h)`

### State

State at window `h` summarizes all information available up to the beginning of that window.

Use interpretable rolling features from the actual MIMIC tables.

#### Static features

From `mimiciv_icu.icustays` and hospital-level joins:

- age
- sex
- first ICU care unit
- ICU admission type if available through admission joins
- comorbidity burden such as Charlson if later added

#### Rolling vital features

From `mimiciv_derived.vitalsign` in the previous `4` hours and previous `8` hours:

- mean heart rate
- max heart rate
- mean MBP
- min MBP
- mean respiratory rate
- max respiratory rate
- min SpO2
- max temperature
- last observed vital value in window

#### Rolling lab features

From `mimiciv_derived.bg`, `mimiciv_derived.chemistry`, and `mimiciv_derived.complete_blood_count`:

- last lactate
- max lactate in prior `8` hours
- delta lactate from previous window
- last pH
- last PaO2/FiO2 ratio
- last creatinine
- delta creatinine
- last bicarbonate
- last sodium
- last potassium
- last WBC
- last platelet
- last hemoglobin

#### Organ support and severity features

From `mimiciv_derived.sofa`, `mimiciv_derived.ventilation`, and later possibly `inputevents`:

- current `sofa_24hours`
- component scores:
  - cardiovascular
  - respiration
  - renal
  - coagulation
  - liver
  - cns
- ventilation status in current window:
  - none
  - supplemental oxygen
  - invasive ventilation
- vasopressor exposure indicator
  - initial proxy from `sofa` cardiovascular component or derived vasopressor rates
- cumulative organ-failure count

#### Trend and trajectory-shape features

- MBP trend over last two windows
- lactate trend over last two windows
- SOFA trend
- number of abnormal features currently active
- number of measurements obtained in past window

#### Missingness features

These are essential in EHR RL.

- was lactate measured in previous `4` hours
- was chemistry measured in previous `4` hours
- was CBC measured in previous `4` hours
- time since last lactate
- time since last creatinine
- time since last WBC

Missingness is part of the state, not just noise.

## Action Design

The action must be observable from logged data and clinically interpretable.

### Recommended first action space

Three actions:

- `0 = continue`
- `1 = intensified_review`
- `2 = escalate`

### Action extraction rule

The action at window `h` is defined from what happens in the next `4` hours, using observable clinical workflow proxies.

#### `continue`

Assign when all of the following hold in `(t_h, t_h+4h]`:

- no new invasive ventilation start
- no new vasopressor start or meaningful vasopressor escalation proxy
- no ICU-level escalation event beyond current baseline
- no urgent repeat sepsis-lab pattern

#### `intensified_review`

Assign when there is increased monitoring or reassessment without major escalation:

- repeat lactate in the next `4` hours
- repeat blood gas in the next `4` hours
- dense follow-up labs after instability
- increased respiratory or hemodynamic checking without new organ-support escalation

This action is meant to capture "the team is worried and is checking again soon."

#### `escalate`

Assign when any of the following occur in the next `4` hours:

- invasive ventilation starts
- vasopressor starts or clear vasopressor escalation occurs
- major organ-support escalation proxy appears
- comparable high-acuity escalation event is observed

### Why this action design works

- it is much closer to a realistic monitoring policy than direct dosing
- it is far easier to extract consistently from MIMIC
- it maps naturally onto support analysis
- it aligns with the pessimism paper:
  - unsupported escalation recommendations are exactly the kind of decisions pessimism should penalize

## Reward Design

Reward should support short-horizon deterioration monitoring.

### Recommended first reward

Use a dense clinical stability reward plus terminal deterioration penalties.

Per-window reward at time `h`:

- negative penalty if:
  - MBP falls below threshold
  - lactate rises substantially
  - SOFA worsens
  - oxygen or ventilation burden increases
  - vasopressor need appears
- small penalty for action burden:
  - unnecessary escalation should not be free

Terminal penalty at `24h`:

- large penalty for:
  - death during or soon after the windowed episode
  - persistent severe organ dysfunction
  - invasive ventilation if newly required
  - vasopressor dependence if newly required

### A concrete first formula

For window `h`, define:

- `r_h = -(0.5 * I[MBP_min < 65])`
- `      -(0.5 * I[lactate >= 2.0])`
- `      -(0.25 * max(sofa_24h_h - sofa_24h_{h-1}, 0))`
- `      -(0.5 * I[new invasive ventilation])`
- `      -(0.5 * I[new vasopressor support])`
- `      -(0.05 * I[action = intensified_review])`
- `      -(0.10 * I[action = escalate])`

Terminal add-on:

- `-2.0` for death in-hospital
- `-1.0` if on invasive ventilation at episode end and not at episode start
- `-1.0` if on vasopressor support at episode end and not at episode start

This is not the only valid reward, but it is concrete, dense, and clinically legible.

### Why not mortality-only

- too sparse
- too delayed
- too sensitive to censoring and downstream care
- too hard to interpret when comparing greedy and pessimistic behavior

Mortality should still be reported as an external outcome, just not used alone as the training reward.

## Real MIMIC Example

Use the deidentified stay:

- `stay_id = 30588857`
- `hadm_id = 24305596`
- care unit:
  - `Cardiac Vascular Intensive Care Unit (CVICU)`
- ICU interval:
  - `2110-01-11 10:16:06` to `2110-01-12 17:17:47`
- sepsis suspicion time:
  - `2110-01-11 12:00:00`
- initial sepsis SOFA time:
  - `2110-01-11 17:00:00`
- sepsis SOFA score:
  - `3`

### What the early windows look like

Around `12:00` to `13:00`:

- heart rate around `93` to `94`
- MBP around `84` to `90`
- respiratory rate around `13` to `14`
- SpO2 around `99` to `100`
- temperature around `35.9` to `36.0`
- lactate `1.1` at `11:50`
- WBC `15.4` at `11:50`
- SOFA 24h score still `0` at `12:00` to `15:00`

Around `16:00`:

- MBP down to `71`
- temperature up to `37.3`
- PaO2/FiO2 ratio down to `280`
- SOFA 24h rises to `3`
- cardiovascular component becomes `1`
- respiration component becomes `2`
- ventilation status:
  - `InvasiveVent` from `13:00` to `20:00`
  - then `SupplementalOxygen`

### How this would map into our task

At `w0`, the state would look moderately abnormal but not yet clearly catastrophic.

At `w1`, the state would show:

- worsening blood pressure
- respiratory support burden
- rising organ dysfunction

In our action space, a plausible logged next-step label here would be:

- `escalate`

This is exactly the kind of case where we want to know:

- would greedy overgeneralize escalation in similar but weakly supported states?
- would pessimism appropriately avoid escalation only when the support is genuinely poor?

## Exact First Experiment

### Objective

Run the ICU-only sepsis analogue of the current linear study.

Compare:

- clinician behavior benchmark
- greedy fitted linear policy
- pessimistic fitted linear policy
- support-masked conservative baseline

### Model class

Stay close to the paper and to our current code:

- linear state-action value model
- ridge regularization
- pessimistic penalty based on feature covariance / uncertainty width

Do not start with a deep sequence model.

### State-action feature map

Construct `phi(s, a)` by concatenating:

- state features
- action one-hot
- selected state-action interactions

Recommended interaction blocks:

- low MBP x escalate
- high lactate x intensified_review
- high SOFA x escalate
- invasive ventilation x continue
- high missingness x intensified_review

This keeps the model linear but expressive enough to behave clinically.

### Fixed-beta study

Use the same experimental structure as before:

- dataset sizes:
  - subsample stays to `500`, `1k`, `2k`, `5k`, `10k`, full cohort
- seeds:
  - multiple patient subsamples
- fixed pessimism:
  - start with `beta = 0.8`

Primary outputs:

- estimated policy value
- support diagnostics
- escalation rate
- disagreement cases between greedy and pessimistic

### Beta sweep

Sweep:

- `beta in {0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0}`

Goal:

- identify whether ICU sepsis has the same turnover as `linear_near_intrinsic`
- detect subgroups where pessimism helps early but becomes too conservative with enough data

## Support Diagnostics

These are essential. In the EHR phase, support analysis is part of the scientific contribution.

### Patient-level support diagnostics

For each decision:

- action frequency in local feature region
- feature novelty score
- chosen-action pessimistic penalty
- whether chosen action falls below support threshold

### Cohort-level support diagnostics

- action frequencies by severity decile
- action frequencies by care unit
- escalation frequency by SOFA band
- escalation frequency by lactate band
- escalation frequency by ventilation status

### Boundary-study subgroups

These are the EHR analogues of `intrinsic` and `near_intrinsic`.

#### Off-support-like subgroup

States where escalation is clinically plausible but rarely logged:

- low-frequency severe combinations
- rare high-SOFA / low-MBP / sparse-lab states
- unusual post-op CVICU sepsis patterns

Expected pattern:

- pessimism should protect against aggressive extrapolation
- no method should convincingly recover unsupported action values

#### Near-support subgroup

States where escalation is rare but present:

- rising SOFA with repeated labs and modest hemodynamic decline
- early respiratory worsening before full shock
- common ICU deterioration patterns with some but not dense escalation history

Expected pattern:

- pessimism helps at low sample size
- fixed positive `beta` eventually becomes too conservative

## Example Cases to Review by Hand

### Case A: Common stable sepsis ICU state

- SOFA low
- MBP stable
- lactate normal
- no new ventilation burden

Desired policy behavior:

- both greedy and pessimistic should usually recommend `continue`

### Case B: Common deteriorating sepsis ICU state

- falling MBP
- moderate SOFA increase
- repeat labs ordered often in historical data

Desired behavior:

- `intensified_review` should be well supported
- pessimism should not suppress it excessively

### Case C: Rare severe instability

- respiratory failure plus hypotension plus sparse matching historical cases

Desired behavior:

- greedy may overreach
- pessimism should expose uncertainty and avoid unsupported escalation claims

### Case D: Rare-support accumulation state

- recurrent warning signs over several windows
- escalation appears in some historical cases but not many

Desired behavior:

- this is the best place to test the EHR analogue of `linear_near_intrinsic`

## Implementation Plan

### Step 1: Build cohort table

Create a cohort script that produces one row per sepsis ICU stay:

- `stay_id`
- `hadm_id`
- `subject_id`
- `t0 = suspected_infection_time`
- ICU timing
- first care unit
- initial sepsis SOFA score

Suggested output:

- `data/ehr/sepsis_icu_cohort.parquet`

### Step 2: Build rolling window table

For each stay and each 4-hour window:

- aggregate vitals
- attach most recent labs
- attach SOFA components
- attach ventilation status
- attach missingness indicators

Suggested output:

- `data/ehr/sepsis_icu_windows.parquet`

### Step 3: Build logged action labels

For each window:

- label `continue`
- `intensified_review`
- or `escalate`

Suggested output:

- `data/ehr/sepsis_icu_actions.parquet`

### Step 4: Build reward table

For each window:

- compute dense reward
- compute terminal outcomes

Suggested output:

- `data/ehr/sepsis_icu_rewards.parquet`

### Step 5: Run first linear offline RL baseline

Use the same code philosophy as the current linear experiments:

- feature-based fitted value estimation
- greedy
- pessimistic
- support-masked

### Step 6: Produce a casebook

For selected stays:

- show trajectory
- show logged action
- show greedy recommendation
- show pessimistic recommendation
- show support diagnostics

Suggested output:

- `docs/ehr_casebook.md`

## Proposed Code Layout

```text
src/peorl/ehr/
  __init__.py
  cohort.py
  windows.py
  actions.py
  rewards.py
  features.py
  ope.py
  experiments.py
```

Suggested scripts:

```text
scripts/
  build_ehr_cohort.py
  build_ehr_windows.py
  inspect_ehr_support.py
  run_ehr_experiment.py
  run_ehr_beta_sweep.py
  make_ehr_casebook.py
```

## Main Risks

### Risk 1: action labels are too noisy

Mitigation:

- keep action space coarse
- validate action frequencies and timing
- chart-review a few examples

### Risk 2: confounding looks like support mismatch

Mitigation:

- include trend and missingness features
- perform subgroup analysis
- use case review, not just aggregate metrics

### Risk 3: reward drives weird behavior

Mitigation:

- use dense reward
- run reward sensitivity checks
- report external outcomes separately

### Risk 4: ICU-only task is still too broad

Mitigation:

- stratify by care unit
- start with a single care unit subset if needed
- use support diagnostics before value claims

## Immediate Recommendation

The first concrete build order should be:

1. cohort extraction from `mimiciv_derived.sepsis3` and `mimiciv_icu.icustays`
2. rolling window builder from `vitalsign`, `bg`, `chemistry`, `complete_blood_count`, `sofa`, and `ventilation`
3. action extraction audit
4. support report
5. first fixed-beta linear EHR baseline

That is the most faithful and defensible way to extend the pessimism paper into ICU-only sepsis using this MIMIC-IV database.

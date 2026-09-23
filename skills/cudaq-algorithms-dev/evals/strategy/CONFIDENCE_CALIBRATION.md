# Proposed confidence calibration and inference specification

## Status and scope

This is a proposal, not an accepted policy and not evidence that the current
calculator has 95% coverage.  Until this specification is reviewed, a later
certification study succeeds, and the frozen caller/manifest wiring is verified,
`statistical_pass` must remain `null` and the result must remain labelled
uncalibrated.  Calibration must use synthetic data only; disclosed or private
evaluation outcomes are not calibration inputs.

The procedure is for generalization to **similar future cases**, not merely
repeat-run inference for this fixed authored suite.  That interpretation needs
the assumptions below; authored prompts are not a random sample merely because
the computation resamples them.

## Estimand and analysis cohort

For metric \(m\), the estimand is

\[
  \theta_m = \operatorname{median}(C_m) /
             \operatorname{median}(B_m),
\]

not the median of within-pair ratios.  `B` and `C` denote baseline and
candidate repetition rows drawn from the scheduled mixture of similar cases.
The schedule is frozen at 22 one-run trigger cases, four one-run execution
cases, and eight three-run execution cases.  Thus the existing pooling gives
22:4:24 repetition-row weight when every run is eligible.  Calibration must
not equalize case weights, average repetitions, change ordering, or otherwise
change that estimand.  Sample medians retain the current even-sample convention
of averaging the two central observations.

Eligibility is selected once: a row enters only when both arms succeeded and
time, total worker tokens, and peak request-input-token proxy are all complete.
Exactly the same selected rows are used for all three metrics.  Metric-specific
filtering is forbidden.  The inferential target is consequently conditional on
joint success and complete measurement.  It is not the efficiency of all
attempted tasks.  Failures, exclusions, and coverage must be reported beside
the bounds; conditioning cannot repair informative failure or measurement
selection.

Generalization requires cases to be independent and exchangeable within each
frozen stratum, the three stratum populations and scheduled proportions to be
representative of future use, and the selection mechanism not to invalidate
the conditional target.  Repetitions within a case may be arbitrarily
dependent.  The fixed baseline-then-candidate order, provider drift, caching,
and shared infrastructure remain confounds.  A calibrated sampling calculation
would not make the arm contrast causal or transport it to materially different
tasks, providers, budgets, or model settings.

## Proposed inference calculation

Freeze the manifest before outcomes.  Its verified SHA-256 supplies the seed;
passing a hash to the API is not itself provenance verification.  Within each
of the three schedule strata, sample the observed number of case IDs with
replacement.  Carry both arms and every jointly eligible repetition of each
sampled case together.  Pool sampled repetition rows exactly as above and
recompute the ratio of arm medians.  Use identical sampled case indices for all
three metrics.

Use 99,999 draws, batch size 2,048, and the current deterministic per-stratum
streams.  The proposed pointwise 95% upper bound is bootstrap order statistic
`ceil(0.95 * (99_999 + 1)) = 95_000`.  No retry, favorable seed selection, or
data-dependent increase in draws is allowed.  Invalid inputs, a missing
stratum, a nonfinite ratio, or absent verified manifest hash produce unknown,
not a pass.  Always report per-stratum case/repetition support, excluded pair
IDs and reasons, distinct bootstrap ratios, bootstrap range, and seed/draw
hashes.

If calibration succeeds and the user adopts this inference rule, the existing
efficiency rule would require, independently for each metric,
`upper_confidence_bound <= 1.001`; all three requirements must hold.  An
observed ratio at or below 1.001 is not sufficient.

## Proposed synthetic scenario envelope

The first experiment has identifier
`paired-case-cluster-percentile-v1-falsification-v1`.  For scenario number `s`
and outer trial `t` (both zero based), initialize the data generator from the
eight consecutive big-endian 32-bit words of
`SHA256("paired-case-cluster-percentile-v1-falsification-v1|outer|s|t")`.
Pass
`SHA256("paired-case-cluster-percentile-v1-falsification-v1|bootstrap|s|t").hexdigest()`
as the synthetic manifest hash.  Cycle the true ratios by `t % 4` through
`[0.995, 0.999, 1.001, 1.005]`; scale every candidate marginal by that ratio.
Coverage is scale-equivariant, while this cycle also exposes decision behavior
around the 0.1% boundary.

Every ordinary trial uses the exact 22/4/8 case schedule and three metrics.
Continuous scenarios generate positive values by exponentiating centered
latent variables.  For triple-run cases, latent repetition value is
`sqrt(kappa) * case_effect + sqrt(1-kappa) * repetition_effect`.  Baseline and
candidate effects have the stated arm correlation `rho`; the joint Gaussian
covariance is the Kronecker product of the 2-by-2 arm correlation matrix and a
3-by-3 metric equicorrelation matrix with off-diagonal 0.5.  Case and
repetition arrays are independent.  Candidate and baseline marginals are
otherwise identical, so their population medians are exactly in the stated
ratio.  Stratum log locations are `(0, 0, 0)` unless specified.  In discrete
scenarios, generate an independent category stream for each metric; “copy”
decisions are independent Bernoulli draws conditional on the stated case
category.  These definitions, including mixture-component coupling, must be
encoded as generator tests before the table can be called frozen.

| s | Proposed scenario |
|---|---|
| 0 | Pairing sanity: log-baseline standard normal with locations `(-1, 0.5, 2)`; candidate is exactly `theta * baseline`; `kappa=0.8`. |
| 1 | Light-tailed, strongly paired: joint normal latents, `rho=0.9`, `kappa=0.5`, metric log-scales `(0.1, 0.3, 0.7)`. |
| 2 | Weak pairing and independent repeats: joint normal latents, `rho=0`, `kappa=0`, log-scale `0.5`. |
| 3 | Strong case clustering: joint normal latents, `rho=0.5`, `kappa=0.95`, log-scale `0.5`. |
| 4 | Heavy tails/outliers: select one mixture component per case, shared by its arms, metrics, and repetitions; conditional paired Gaussian draws use `rho=0.7`, `kappa=0.8`, and scale 0.3 with probability 0.98 or scale 3 with probability 0.02. |
| 5 | Stable ties: values `(0.5, 1, 2)` with probabilities `(0.25, 0.50, 0.25)`; candidate copies the baseline category with probability 0.8 and otherwise takes an independent category; triple repetitions copy the case category with probability 0.8. |
| 6 | Near-discontinuous median: values `(0.5, 1, 2)` with probabilities `(0.49, 0.02, 0.49)`, using the same dependence construction as scenario 5.  The population median is unique but weakly identified. |
| 7 | Stratum mixture plus common selection: scenario 1 with locations `(-1, 0.5, 2)`; retain a repetition row only when every standardized baseline and candidate latent for all three metrics has absolute value at most 2.  This symmetric rule preserves the known conditional medians and enforces one joint cohort. |

Scenario 8 is a diagnostic-only sparse variant of scenario 1 retaining just
the first generated case in each stratum (and all its repetitions).  It must be
reported, but it cannot help the method pass calibration.  It tests whether a
nominal bound can look reassuring with essentially no task diversity.  Whether
such sparse support must mechanically return unknown is a policy/API choice;
the current implementation only warns, so this document does not silently add
an acceptance minimum.

Before execution, review a standalone generator specification and tests showing
the 22/4/8 shape, marginal probabilities, known median ratios, dependence
parameters, and common selection.  This document does not authorize that code
or a run.

## Bounded first experiment: falsification only

Run 64 outer trials for each of scenarios 0--7 (512 ordinary calculator calls)
and 64 sparse diagnostics, all with the production 99,999 draws.  Do not replace
failed trials or stop early.  Repeat ordinary trials `t=0..3` at batch sizes 1
and 8,192 and require identical bounds and draw hashes.  The hard maximum is 576
ordinary/sparse calls plus 64 determinism calls, about 64 million streamed
bootstrap replicates.  Cap execution at two CPU-hours and 2 GiB resident
memory.  Exceeding a cap makes the pilot incomplete.  Record wall time,
hardware, Python/NumPy versions, code hash, every unknown, and every exception.

For each scenario and metric, report the number of informative bounds, the
fraction with `upper_bound >= theta`, and the exact binomial interval.  The
pilot refuses the present method/version if scenario 0's point ratio or bound
differs from `theta` by more than `8 * float64_epsilon * max(1, abs(theta))`
(roundoff allowance only), a complete-data scenario returns
unknown, any determinism/finite-JSON check fails, scenario 7 has fewer than 48
informative trials, any cell's non-covering fraction exceeds 10%, or a resource
cap is exceeded.  Dropping a scenario, changing seeds, or rerunning until
favorable is forbidden.  The 90% screen is intentionally coarse, not a proof
threshold.

No result from this pilot will be treated as certifying 95% coverage. Passing only means
the method survives this inexpensive falsification screen.  A later
certification study needs separate approval and a new, fresh seed namespace.
Its sample size must be fixed from desired Monte Carlo precision and measured
pilot runtime, without using pilot coverage to tune the method or scenario
set.  It must predeclare a simultaneous Monte Carlo validation rule across all
scenario/metric cells—for example, exact binomial lower bounds with familywise
error allocation—and require every lower bound to be at least 0.95.  Until that
later specification and run succeed, coverage remains uncalibrated and
`statistical_pass` remains `null`.  This separation avoids presenting a very
large simulation campaign as a prerequisite to current diagnostic progress.

Also report bound excess (`upper_bound - theta`), probability of satisfying
the 1.001 rule at each cycled truth, and sparse-scenario behavior.  These are
precision/power diagnostics, not gates: the user has not chosen a minimum
power.  A method can have adequate coverage yet be too imprecise to establish a
0.1% margin.

## Pointwise bounds and the all-metric decision

The current three 95% bounds are **pointwise**.  Requiring all three limits does
not make them a simultaneous 95% confidence family; the probability that all
three intervals cover can be below 95%.  That is distinct from the accepted
all-metric noninferiority decision.  Write each null as
`H0_m: theta_m > 1.001` (material regression beyond the allowed ceiling) and
reject it only when its valid one-sided 95% upper bound is at most 1.001.
This strict null respects the user's inclusive ceiling; for a true null,
passing implies an upper bound strictly below the true ratio. A false joint pass (all three nulls rejected while at
least one is true) is a subset of falsely rejecting any one true component
null.  Therefore valid level-5% component tests make this intersection--union
decision level at most 5%, regardless of dependence; Bonferroni tightening is
not required for that decision.

Simultaneous 95% coverage of all three displayed intervals would be a separate
reporting objective and would require a different construction.  The user has
not selected that objective, so this proposal neither imposes it nor relaxes
the selected independent-per-metric rule.  Genuine remaining choices are
whether sparse strata should mechanically return unknown and whether to demand
a minimum power/precision before funding a later certification study; neither
choice is silently made here.

Synthetic coverage establishes behavior only over the frozen scenario
envelope.  It cannot prove coverage for every heavy-tailed, tied, selected, or
dependent task population, validate the similarity of authored cases to future
tasks, remove provider/order effects, or turn conditional complete-case
efficiency into unconditional efficiency.

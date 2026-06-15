# Assumptions And Known Limits

## Uncertainty Calibration

Per-temperature-group conformal prediction achieves the nominal 80% coverage target for room and hot cohorts (empirical: 100% each) but falls short for the cold cohort under LOBO calibration (74.5% vs 80% target). This under-coverage for cold batteries is a documented finding caused by thermal distribution shift — it is presented as a result, not hidden as a bug.

## Dataset Name

Batteries B0005–B0056 are from the **NASA PCoE (Prognostics Center of Excellence)** dataset (Saha & Goebel, 2007), **not** the CALCE dataset (University of Maryland). These are distinct benchmarks. References to "NASA CALCE" in older documentation scripts are incorrect; the correct citation is "NASA PCoE".

## Survival Module

The survival/hazard module currently observes ≤5 events on the 6 test batteries. Hazard coefficient estimates are statistically unreliable below 10 events. Survival risk scores should be treated as exploratory. See `WEAKNESSES_FIXES.md` for the event-redefinition plan.

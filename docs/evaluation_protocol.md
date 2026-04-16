# Evaluation Protocol
<!-- Updated: 2026-04-16 -->

This document defines the non-negotiable evaluation rules for the paper. All results must follow this protocol. Do not deviate without updating this file.

---

## Unit of Splitting

**Battery-level split only. Never cycle-level random split across the same battery.**

### Reason

Cycle-level leakage makes performance unrealistically optimistic. If cycles from the same battery appear in both train and test, the model has seen the degradation trajectory during training and is not genuinely generalizing.

---

## Main Evaluation Protocol

**Leave-one-battery-out cross-validation (LOBO-CV) or grouped battery-level folds (GroupKFold with `groups=battery_id`).**

- Minimum 5 folds where battery count allows.
- Every fold: no battery ID appears in both train and validation.
- Report: mean RMSE ± std across folds.

---

## Secondary Protocol

A held-out battery test set, used **only** for final reported numbers:

- Selected before any modeling begins.
- Never touched during hyperparameter tuning or threshold selection.
- Used exactly once for final Tables 2–4.

---

## Split Integrity Rules

For each sample in the processed data, the following columns must be present:

| Column | Purpose |
|--------|---------|
| `battery_id` | Grouping key — never split within |
| `source_dataset` | NASA or CALCE |
| `temperature_group` | room / hot / cold |
| `cycle_index` | Temporal ordering |
| `RUL` | Prediction target |

### Checks that must pass before modeling:

1. No `battery_id` appears in both train and test sets.
2. No `battery_id` appears in both train and calibration sets.
3. No leakage through scaling (scaler fitted on train only, applied to val/test).
4. No leakage through conformal calibration (calibration batteries are a separate held-out set).

Run `src/verification/check_group_split.py` before every pipeline run. If any overlap count is non-zero, **stop and fix the split**.

---

## CALCE Evaluation Policy

CS2_38 alone as the single CALCE test battery is too fragile. Use one of:

### Preferred

Leave-one-battery-out across CALCE batteries: train on 3, test on 1, rotate four times. Report per-rotation RMSE and mean.

### Acceptable fallback

Grouped cross-validation across all 38 batteries. Always report:
- Per-dataset performance (NASA vs CALCE separately)
- Per-battery performance (per_battery_results.csv)

---

## Official Reported Metrics

### Prediction Quality (Table 2)

| Metric | Definition |
|--------|-----------|
| RMSE | Root mean squared error on RUL, battery-level grouped CV |
| MAE | Mean absolute error on RUL |

Report separately for: Statistical baseline, XGBoost, TCN.

### Uncertainty Quality (Table 3)

| Metric | Definition |
|--------|-----------|
| Empirical coverage | Fraction of true RUL values inside the predicted interval |
| Average interval width | Mean (upper − lower) across all test observations |
| Calibration score | Agreement between nominal and empirical coverage |

Report separately for: room / hot / cold temperature groups, and per dataset.

### Risk Quality (Table 4 — partial)

| Metric | Definition |
|--------|-----------|
| AUC | Area under ROC for failure-within-20-cycles prediction |
| Precision / Recall | At operating threshold from validation set |

Fixed horizon: **20 cycles** (consistent with `configs/pipeline.yaml → risk.horizon_cycles`).

### System Reliability Quality (Table 4 — core result)

| Metric | Definition |
|--------|-----------|
| PASS rate | Fraction of batteries / observations classified PASS |
| WARNING rate | Fraction classified WARNING |
| FAIL rate | Fraction classified FAIL |
| RMSE (PASS only) | Prediction error on PASS-labelled subset |
| RMSE (WARNING only) | Prediction error on WARNING-labelled subset |
| RMSE (FAIL only) | Prediction error on FAIL-labelled subset |
| Severe-error catch rate | Fraction of large errors (e.g. >100 cycles) assigned WARNING or FAIL |

**Expected pattern:** PASS should have lowest error and lowest failure rate. FAIL should have highest error and highest near-term failure rate. If this pattern does not hold, the audit rules are wrong.

---

## Threshold Selection Policy

**Do not pick thresholds by looking at the test set.**

| Threshold | How to set |
|-----------|-----------|
| Uncertainty width (PASS/WARN/FAIL) | 75th / 90th percentile of validation interval widths |
| Risk threshold | Validation operating point from survival predictions |
| Anomaly threshold | Top-5% error on validation set (not test) |
| Battery-level aggregation window | Last K=10 observed cycles (frozen in config) |

All thresholds must be recorded in `configs/pipeline.yaml` under the `audit` section before running experiments on the test set.

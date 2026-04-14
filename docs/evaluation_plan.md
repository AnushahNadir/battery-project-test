# Evaluation Plan

## Objective
Estimate generalization performance for battery RUL prediction without cycle-level leakage.

## Leakage Policy
- Unit of independence: `battery_id`.
- Rule: cycles from the same `battery_id` must never appear in both train and evaluation partitions.
- Enforced in pipeline:
  - Holdout split by battery in `src/modeling/run_full_pipeline.py`.
  - Explicit overlap checks between train/calibration/test battery sets.
  - Grouped cross-validation with `GroupKFold(groups=battery_id)`.

## Holdout Evaluation
- Primary report uses a battery-level holdout split:
  - Training batteries.
  - Calibration batteries (subset of train for conformal calibration).
  - Test batteries (never seen during model fit).
- Metrics are computed on test batteries only.

## Cross-Validation (Battery-level)
- Strategy: `GroupKFold`.
- Group key: `battery_id`.
- Default folds: 5 (automatically reduced when battery count is smaller).
- Current implementation trains a fresh XGBoost model per fold and reports:
  - RMSE.
  - MAE.
  - Fold battery counts (train/validation).
- Artifacts:
  - `data/processed/modeling/groupkfold_cv_report.json`
  - `trained_models/model_metrics.json` under `groupkfold_cv`

## Why GroupKFold
Random row-level CV leaks battery trajectory information across folds and can inflate reported accuracy. Grouped folds provide a more realistic estimate of performance on unseen batteries.

## Acceptance Criteria
- Zero `battery_id` overlap in every fold.
- Zero overlap between holdout train and test battery sets.
- `groupkfold_cv_report.json` generated on each pipeline run.

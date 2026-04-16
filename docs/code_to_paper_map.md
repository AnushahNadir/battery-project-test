# Code-to-Paper Map
<!-- Updated: 2026-04-16 -->

Maps every paper section to the implementation files that support it.

---

## Section 3.1 — Data Preprocessing

**Paper claim:** Standardized cycle-level features extracted from raw NASA and CALCE data.

| Artifact | File / Path |
|----------|-------------|
| Entry point | `src/main.py` |
| Pipeline runner | `src/pipeline/run_analysis` |
| CALCE loader | `src/pipeline/calce_loader.py` |
| Combined dataset builder | `scripts/build_combined_dataset.py` |
| Output: cycle features | `data/processed/cycle_features_with_rul.csv` |
| Output: QC report | `data/processed/qc_report.json` |
| Output: validation report | `data/processed/validation_report.json` |
| Output: confidence report | `data/processed/confidence_report.json` |

**Paper belongs:** Section 3.1, Table 1 (dataset summary)

---

## Section 3.2 — Predictive Modeling

**Paper claim:** Three predictors of increasing complexity compared at battery level.

| Artifact | File |
|----------|------|
| XGBoost (primary) | `src/modeling/ml_model.py` |
| TCN (sequence) | `src/modeling/dl_sequence_model.py` |
| Statistical baseline | `src/modeling/statistical_baseline.py` |
| Pipeline orchestration | `src/modeling/run_full_pipeline.py` |
| Config (hyperparams) | `configs/pipeline.yaml` → `xgboost`, `dl_model` |

**Paper belongs:** Section 3.2, Table 2 (predictive model comparison)

---

## Section 3.3 — Uncertainty Quantification

**Paper claim:** Conformal prediction wraps XGBoost to produce calibrated 90% confidence intervals, stratified by temperature group.

| Artifact | File |
|----------|------|
| Conformal calibrator | `src/modeling/conformal.py` |
| Uncertainty analysis | `src/modeling/uncertainty.py` |
| Config (coverage target, safety factor) | `configs/pipeline.yaml` → `conformal`, `uncertainty` |
| Output: uncertainty estimates | `data/processed/modeling/uncertainty_estimates.json` |
| Output: conformal coverage report | `data/processed/modeling/conformal_coverage_report.json` |
| Output: uncertainty metrics | `data/processed/modeling/uncertainty_metrics.json` |

**Paper belongs:** Section 3.3, Table 3 (uncertainty calibration results)

---

## Section 3.4 — Risk Estimation

**Paper claim:** Per-cycle probability of failure within the next 20 cycles, derived from a survival/hazard model.

| Artifact | File |
|----------|------|
| Survival risk runner | `src/analysis/survival_risk.py` |
| Config (horizon, thresholds) | `configs/pipeline.yaml` → `risk`, `survival` |
| Output: survival predictions | `data/processed/modeling/survival_risk_predictions.csv` |
| Output: survival metrics | `data/processed/modeling/survival_risk_metrics.json` |
| Output: survival report | `data/processed/modeling/survival_risk_report.md` |

**Paper belongs:** Section 3.4, Table 4 (audit layer — risk column)

---

## Section 3.5 — Reliability Audit Layer

**Paper claim:** Deterministic per-battery verdict (PASS / WARNING / FAIL) derived from uncertainty width, failure risk, anomaly flag, and consistency checks.

| Artifact | File |
|----------|------|
| Supervisor review | `src/modeling/supervisor_review.py` |
| Analysis planner | `src/supervisor/analysis_planner.py` |
| Analysis rules | `src/supervisor/analysis_rules.py` |
| Consistency check | `src/verification/consistency_check.py` |
| Uncertainty assessment | `src/verification/uncertainty_assessment.py` |
| Group split checker | `src/verification/check_group_split.py` |
| Config (thresholds) | `configs/pipeline.yaml` → `supervisor`, `gating` |
| Output: final report | `data/processed/modeling/final_system_report.md` |
| Output: per-battery results | `data/processed/modeling/per_battery_results.csv` *(to be created)* |

**Paper belongs:** Section 3.5, Table 4 (audit layer results by PASS/WARNING/FAIL)

---

## Section 4 — Experiments

**Paper claim:** Five experiments: (1) predictive baselines, (2) uncertainty calibration, (3) reliability stratification, (4) audit layer evaluation, (5) ablation.

| Artifact | File |
|----------|------|
| Full pipeline runner | `src/modeling/run_full_pipeline.py` |
| GroupKFold CV | `src/modeling/ml_model.py` → `evaluate_groupkfold_cv` |
| Ablation runner | *(to be created)* `scripts/run_ablation.py` |
| Config (split seed, CV) | `configs/pipeline.yaml` → `split` |
| Output: metrics prediction | `data/processed/modeling/metrics_prediction.json` *(to be created)* |
| Output: metrics uncertainty | `data/processed/modeling/metrics_uncertainty.json` *(to be created)* |
| Output: metrics risk | `data/processed/modeling/metrics_risk.json` *(to be created)* |
| Output: metrics audit | `data/processed/modeling/metrics_audit.json` *(to be created)* |
| Output: ablation results | `data/processed/modeling/ablation_results.csv` *(to be created)* |

---

## run_full_pipeline.py Block-by-Block Analysis

| Block | Input | Transformation | Output | Paper? |
|-------|-------|----------------|--------|--------|
| Config load | `configs/pipeline.yaml` | Parse YAML | cfg object | Appendix (reproducibility) |
| Battery split | `cycle_features_with_rul.csv` | Stratified by temp group | train/cal/test lists | Section 4 |
| Overlap check | train/cal/test lists | Assert disjoint | sys.exit if leak | Section 4 (methodology) |
| Pre-train anomaly filter | train_df | Flag known bad batteries | cleaned train_df | Appendix |
| Statistical baseline | train_df, test_df | Trend extrapolation | stat_metrics | Section 3.2 / Table 2 |
| XGBoost | train_df, test_df | Gradient boosting on cycle features | ml_metrics, ml_model | Section 3.2 / Table 2 |
| TCN | train_df, test_df | Temporal conv on sequences | dl_metrics, dl_model | Section 3.2 / Table 2 |
| GroupKFold CV | train_df | 5-fold battery-grouped CV | cv_metrics | Section 4 / Table 2 |
| Conformal calibration | cal_df, ml_model | LOBO per temp group | calibrator | Section 3.3 / Table 3 |
| Drift monitor | train_df vs test_df | PSI per feature | drift_report | Appendix |
| Uncertainty analysis | full df + models | Ensemble RUL + CI | uncertainty_estimates.json | Section 3.3 |
| Survival/hazard risk | test_df | Kaplan-Meier per battery | survival_risk_predictions.csv | Section 3.4 |
| Anomaly detection | test_df, ml_model | Top-5% error flag | anomalies.json | Section 3.5 |
| Reasoning | test_df, ml_model | Hypotheses + counterfactuals | degradation_hypotheses.json | Appendix only |
| Supervisor review | modeling outputs | Stage-by-stage verdict | final_system_report.md | Section 3.5 |
| Per-battery results | all outputs | Battery-level aggregation | per_battery_results.csv | Section 4 / Tables |

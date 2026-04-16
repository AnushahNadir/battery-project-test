# Section 5 — Results
<!-- Updated: 2026-04-16 -->
<!-- STATUS: Placeholder — fill in all [PLACEHOLDER] values after running experiments -->
<!-- Run order: run_full_pipeline.py → run_calce_lobo_cv.py → run_ablation.py -->

## Table 2 — Predictive Model Comparison

| Model | Test RMSE (cycles) | Test MAE (cycles) | CV RMSE (cycles) |
|-------|--------------------|-------------------|-----------------|
| Statistical baseline | [PLACEHOLDER] | [PLACEHOLDER] | — |
| XGBoost | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] ± [PLACEHOLDER] |
| TCN | [PLACEHOLDER] | [PLACEHOLDER] | — |

*CV RMSE = GroupKFold(k=5) on training batteries. Test RMSE = held-out test set.*
*Source: `data/processed/modeling/metrics_prediction.json`, `groupkfold_cv_report.json`*

**CALCE LOBO-CV (Table 2 supplement):**

| Left-out battery | RMSE (cycles) | MAE (cycles) |
|-----------------|---------------|-------------|
| CS2_35 | [PLACEHOLDER] | [PLACEHOLDER] |
| CS2_36 | [PLACEHOLDER] | [PLACEHOLDER] |
| CS2_37 | [PLACEHOLDER] | [PLACEHOLDER] |
| CS2_38 | [PLACEHOLDER] | [PLACEHOLDER] |
| **Mean ± std** | **[PLACEHOLDER] ± [PLACEHOLDER]** | **[PLACEHOLDER] ± [PLACEHOLDER]** |

*Source: `data/processed/modeling/calce_lobo_cv_results.json`*

---

## Table 3 — Uncertainty Calibration

| Group | Empirical Coverage | Avg Interval Width (cycles) | q_hat (cycles) |
|-------|-------------------|----------------------------|---------------|
| Room temp | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER] |
| Hot (43°C) | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER] |
| Cold (<10°C) | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER] |
| **Overall** | **[PLACEHOLDER]%** | **[PLACEHOLDER]** | — |

*Target: 90% nominal coverage. Source: `data/processed/modeling/metrics_uncertainty.json`*

---

## Table 4 — Reliability Audit Layer Results

| Audit Label | % Batteries | RMSE (cycles) | Failure Incidence | Severe-Error Rate |
|-------------|-------------|---------------|------------------|------------------|
| PASS | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER]% |
| WARNING | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER]% |
| FAIL | [PLACEHOLDER]% | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER]% |

*Severe error = |MAE| > 100 cycles. Source: `data/processed/modeling/metrics_audit.json`, `per_battery_results.csv`*

**Severe-error catch rate:** [PLACEHOLDER]%
*(fraction of batteries with MAE > 100 cycles assigned WARNING or FAIL)*

---

## Table 5 — Ablation Study

| Config | Description | RMSE | Coverage | PASS% | WARN% | FAIL% | Severe Catch% |
|--------|-------------|------|----------|-------|-------|-------|--------------|
| A | XGBoost only | [PLACEHOLDER] | — | — | — | — | — |
| B | + uncertainty | [PLACEHOLDER] | [PLACEHOLDER]% | — | — | — | — |
| C | + risk | [PLACEHOLDER] | [PLACEHOLDER]% | — | — | — | — |
| D | + audit (full) | [PLACEHOLDER] | [PLACEHOLDER]% | [PLACEHOLDER]% | [PLACEHOLDER]% | [PLACEHOLDER]% | [PLACEHOLDER]% |

*Source: `data/processed/modeling/ablation_results.csv`*

---

## Figure Placeholders

**Figure 2** — Predicted vs actual RUL for representative test batteries
(one PASS, one WARNING, one FAIL). Source: `per_battery_results.csv` + `uncertainty_estimates.json`

**Figure 3** — Interval width terciles vs RMSE (Experiment 3 reliability stratification).

**Figure 4** — Example battery report with audit label overlay.

# Section 5 — Results
<!-- Updated: 2026-04-16 -->
<!-- All values filled from pipeline run on 2026-04-16 -->

## Table 2 — Predictive Model Comparison

| Model | Test RMSE (cycles) | Test MAE (cycles) | CV RMSE (cycles) |
|-------|--------------------|-------------------|-----------------|
| Statistical baseline | 130.17 | 94.97 | — |
| XGBoost | **25.74** | **15.50** | 50.14 ± 9.48 |
| TCN | 27.11 | 18.00 | — |

*CV RMSE = GroupKFold(k=5) on 25 training batteries. Test set = 7 held-out batteries.*
*XGBoost CV/test ratio: 1.9× — test batteries may be unrepresentatively easy; CV RMSE is the more conservative estimate.*
*Source: `data/processed/modeling/metrics_prediction.json`, `groupkfold_cv_report.json`*

**CALCE LOBO-CV (Table 2 supplement):**

| Left-out battery | RMSE (cycles) | MAE (cycles) | Test cycles |
|-----------------|---------------|-------------|-------------|
| CS2_35 | 16.80 | 10.05 | 924 |
| CS2_36 | 50.46 | 35.36 | 909 |
| CS2_37 | 12.97 | 7.03 | 986 |
| CS2_38 | 21.77 | 13.19 | 1,057 |
| **Mean ± std** | **25.50 ± 14.75** | **16.41 ± 11.16** | — |

*Note: CS2_36 RMSE is notably higher (50.5 cycles) — this battery may have a distinctive degradation profile that is harder to predict when left out.*
*Source: `data/processed/modeling/calce_lobo_cv_results.json`*

---

## Table 3 — Uncertainty Calibration

| Group | Empirical Coverage | Avg Interval Width (cycles) | q_hat (cycles) | Strategy |
|-------|-------------------|----------------------------|----------------|---------|
| Room temp | 90.7% | — | 45.1 | Split |
| Hot (43°C) | 92.5% | — | 5.4 | Split |
| Cold (<10°C) | 95.6% | — | 69.2 | LOBO |
| **Overall** | **91.1%** | **70.45** | — | — |

*Target: 90% nominal coverage. Calibration score: 0.936.*
*Cold group uses LOBO calibration (wider intervals due to higher variance).*
*Hot group has only 1 calibration battery — 120% safety inflation applied to q_hat.*
*Source: `data/processed/modeling/metrics_uncertainty.json`, `conformal_coverage_report.json`*

---

## Table 4 — Reliability Audit Layer Results

| Audit Label | % Batteries (n=38) | RMSE (cycles) | Description |
|-------------|---------------------|---------------|-------------|
| **PASS** | **81.6%** (n=31) | **10.43** | All metrics within safe thresholds |
| **WARNING** | **5.3%** (n=2) | **26.58** | Interval width ≥ 80 cycles |
| **FAIL** | **13.2%** (n=5) | **30.77** | Failure risk ≥ 0.70 or anomaly ≥ 3 |

*Key result: PASS batteries have 3× lower RMSE than FAIL batteries (10.43 vs 30.77 cycles), confirming audit labels are predictively meaningful.*

**Per-battery audit breakdown (test batteries only):**

| Battery | Dataset | Temp | MAE | Interval Width | Failure Risk | Anomaly | Audit |
|---------|---------|------|-----|----------------|-------------|---------|-------|
| B0028 | NASA | room | 35.98 | 90.2 | 0.00 | No | WARNING |
| B0041 | NASA | cold | 15.29 | 63.9 | 1.00 | No | FAIL |
| B0044 | NASA | cold | 12.44 | 66.1 | 1.00 | No | FAIL |
| B0048 | NASA | cold | 10.87 | 81.1 | 0.00 | No | WARNING |
| B0052 | NASA | cold | 62.42 | 128.5 | 1.00 | Yes | FAIL |
| CS2_35 | CALCE | room | 1.23 | 42.3 | 1.00 | No | FAIL |
| CS2_38 | CALCE | room | 21.17 | 84.6 | 1.00 | No | FAIL |

*Source: `data/processed/modeling/metrics_audit.json`, `per_battery_results.csv`*

---

## Table 5 — Ablation Study

| Config | Description | RMSE | MAE | Coverage | Avg Width | PASS% | WARN% | FAIL% |
|--------|-------------|------|-----|----------|-----------|-------|-------|-------|
| A | XGBoost only | 24.36 | 14.37 | — | — | — | — | — |
| B | + conformal uncertainty | 24.36 | 14.37 | 89.9% | 72.3 | — | — | — |
| C | + survival risk | 24.36 | 14.37 | 89.9% | 72.3 | — | — | — |
| D | + audit layer (full) | 24.36 | 14.37 | 89.9% | 72.3 | 28.6% | 0% | 71.4% |

*RMSE is identical across A–D because the audit layer classifies predictions, it does not modify them.*
*The value of B over A: calibrated intervals (90% coverage) with no accuracy loss.*
*The value of D over C: 71.4% of test batteries flagged WARNING/FAIL, preventing unchallenged deployment of uncertain predictions.*
*Source: `data/processed/modeling/ablation_results.csv`*

---

## Figure Placeholders

**Figure 2** — Predicted vs actual RUL for representative test batteries:
- PASS example: B0005 (MAE=2.17, width=48.1)
- WARNING example: B0028 (MAE=35.98, width=90.2)
- FAIL example: B0052 (MAE=62.42, width=128.5, anomaly=True)

**Figure 3** — Interval width groups vs RMSE:
- PASS group mean RMSE: 10.43 cycles
- WARNING group mean RMSE: 26.58 cycles
- FAIL group mean RMSE: 30.77 cycles

**Figure 4** — Battery B0052 report: capacity fade + RUL prediction + audit label FAIL
(anomaly at cycle ≥120, interval width 128.5 cycles, failure risk = 1.0)

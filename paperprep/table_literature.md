# Table T10: Comparison with Prior Work

**Important caveat (must appear before table in paper):**
Direct RMSE comparison is misleading because the tasks differ fundamentally.
Within-battery methods (Zhao 2022, Catelani 2021) observe early cycles of the *same* target
battery and extrapolate forward — the target cell's degradation curve is already partially
visible. Cross-battery methods (this work) generalise to *entirely unseen* cells with no
prior observations. The latter is substantially harder and produces higher RMSE by design.

| Paper | Dataset | Task type | EOL definition | Reported error | Uncertainty | Multi-seed |
|-------|---------|-----------|---------------|----------------|-------------|------------|
| Lin et al. 2023 (GRU+HMM) | Oxford + NASA Randomised | SOH curve fitting | SOH % threshold | RMSE 0.2–1.8% SOH | HMM residual bounds | No |
| Zhao et al. 2022 (BLS-LSTM) | NASA B0005/B0006 + CALCE CX2 | **Within-battery** RUL | 70% of nominal | AE ≈ 1 cycle | None | No |
| Catelani et al. 2021 (ESN) | NASA B0005–B0018 | **Within-battery** RUL | 70% of nominal | EE 0–6 cycles | t-distribution bounds | No |
| **This work** (XGBoost + TCN ensemble) | NASA PCoE — 34 cells, 3 thermal groups | **Cross-battery** RUL | 80% of per-cell measured init cap (range 0.81–1.53 Ah) | RMSE 44.74 ± 8.38 cycles (5 seeds) | Split-conformal per temperature group | **Yes (5 seeds)** |

## Key differentiators of this work

| Dimension | Prior work | This work |
|-----------|-----------|-----------|
| Prediction target | Within-battery (target cell partially observed) | Cross-battery (target cell entirely unseen) |
| Uncertainty method | None or parametric bounds | Split-conformal prediction (coverage guarantee) |
| Per-group calibration | Not reported | LOBO for cold; split for room/hot |
| Coverage failure analysis | Not reported | Mechanistically linked to PSI drift (5 RED features) |
| Reproducibility | Single seed | 5 seeds with ± std reported |
| Dataset scope | 2–14 cells | 34 cells, 3 thermal cohorts |

## Notes on cited papers

**Lin et al. 2023** addresses state-of-health (SOH) estimation, not RUL prediction —
the target variable and task structure differ. Included for context only.

**Zhao et al. 2022** uses BLS (Broad Learning System) + LSTM on a within-battery setup.
The sub-1-cycle AE figures rely on early-cycle observations of the target battery being available.

**Catelani et al. 2021** uses an Echo State Network on NASA B0005–B0018 (room temperature only,
18 cells). Within-battery approach; 70% EOL definition; uncertainty via t-distribution residuals.
The 0–6 cycle end-of-prediction error is measured near EOL on the same battery used for training
trajectory fitting.

**This work** uses none of the target battery's cycles for training or trajectory fitting.
All 6 test batteries (B0030, B0033, B0036, B0041, B0044, B0052) are entirely held out.
The higher RMSE (44.74 ± 8.38 cycles) is the honest cost of the harder, more realistic task.

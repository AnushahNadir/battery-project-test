# Table T2: Dataset Summary by Temperature Group

| Group | N cells | Cycles (mean) | Init cap mean (Ah) | EOL threshold mean (Ah) | Cells reaching EOL | Split role |
|-------|---------|--------------|-------------------|------------------------|-------------------|------------|
| cold | 16 | 72 | 0.975 | 1.041 | 16/16 | train/cal/test |
| room | 11 | 122 | 1.711 | 1.404 | 7/11 | train/cal/test |
| hot | 7 | 43 | 1.746 | 1.408 | 3/7 | train/cal/test |
| **Total** | **34** | **82** | **1.372** | **1.234** | **26/34** | — |

*Split (seed 42): 22 train, 6 calibration, 6 test.*
*Test batteries: B0030, B0033, B0036, B0041, B0044, B0052.*


# Table T8: Conformal Interval Width by Temperature Group

Conformal CI width = 2 x q_hat (the calibration quantile per group).
Source: conformal_coverage_report.json

| Group | Strategy | q_hat (cycles) | CI width = 2*q_hat | Test coverage | Comment |
|-------|---------|---------------|-------------------|--------------|---------|
| room  | split conformal | 69.71 | 139.42 | 100.0% | Over-wide; excessively conservative |
| hot   | split conformal | 25.28 |  50.56 | 100.0% | Reasonable width; over-covers |
| cold  | LOBO            | 31.98 |  63.96 |  74.5% | Below 80% target despite LOBO |
| **Overall** | mixed | — | — | **91.85%** | Mean uncertainty width: **60.62 cycles** (uncertainty_metrics.json) |

Note: The large room-group CI (139.42 cycles) dominates the overall mean.
LOBO calibration for cold batteries narrows q_hat vs split conformal but still
cannot achieve 80% coverage because PSI drift (1.01-2.57 on 5 features)
violates the exchangeability assumption.
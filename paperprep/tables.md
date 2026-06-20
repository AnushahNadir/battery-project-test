# Table T3: Model RMSE Across 5 Seeds

| Model | Seed 42 RMSE | Mean (5 seeds) | Std (5 seeds) |
|-------|-------------|----------------|---------------|
| XGBoost | 21.74 | 44.74 | 8.38 |
| TCN/MLP (DL) | 27.35 | 41.34 | 4.78 |
| Statistical baseline | ~45.64 | 319.66 | 196.58 (auto-excluded from ensemble) |
| Ensemble (ml=0.557, dl=0.443) | -- | Not computed per-seed | -- |

# Table T4: Per-Battery XGBoost RMSE (Seed 42)

| Battery | Temp group | XGBoost RMSE | DL RMSE |
|---------|-----------|-------------|---------|
| B0030 | hot | 1.93 | 9.31 |
| B0033 | room | 5.88 | 7.79 |
| B0036 | room | 15.44 | 35.56 |
| B0041 | cold | 20.80 | 33.00 |
| B0044 | cold | 32.26 | 29.67 |
| B0052 | cold | 63.89 | 40.63 |

# Table T5: XGBoost Feature Importance (Seed 42)

| Rank | Feature | Importance | Direction |
|------|---------|-----------|-----------|
| 1 | temp_mean | 0.276 | negative |
| 2 | i_mean | 0.196 | negative |
| 3 | cycle_index | 0.121 | negative |
| 4 | temp_max | 0.107 | negative |
| 5 | duration_s | 0.098 | negative |
| 6 | ah_est | 0.058 | negative |
| 7 | capacity | 0.048 | negative |
| 8 | i_min | 0.036 | negative |
| 9 | energy_j | 0.031 | negative |
| 10 | v_mean | 0.031 | negative |

# Table T6: Feature Ablation Results

| Configuration | Mean RMSE | Std | Delta vs full |
|---------------|----------|-----|--------------|
| Full (10 features) | 22.01 | 8.98 | -- |
| Remove cycle_index | 29.11 | 12.20 | +7.10 |
| Remove capacity | 22.30 | 9.17 | +0.29 |

# Table T7: Per-Group Conformal Coverage

| Temp group | Strategy | q_hat (cycles) | Empirical coverage | vs 80% target |
|-----------|---------|---------------|-------------------|--------------|
| room | split | 69.71 | 100.0% | +20.0% |
| hot | split | 25.28 | 100.0% | +20.0% |
| cold | lobo | 31.98 | 74.5% | -5.5% |
| Overall | mixed | -- | 91.85% | +11.85% |

# Table T9: Distribution Shift (PSI) -- Drift Report

| Feature | PSI | Status | Train mean | Test mean |
|---------|-----|--------|-----------|----------|
| i_mean | 2.573 | RED | 1.610 | -- |
| temp_max | 2.462 | RED | 34.615 | -- |
| v_mean | 2.077 | RED | 1.914 | -- |
| duration_s | 1.814 | RED | 3175.078 | -- |
| temp_mean | 1.010 | RED | 26.912 | -- |
| capacity | 0.914 | RED | 1.338 | -- |
| ah_est | 0.876 | RED | 1.226 | -- |
| energy_j | 0.874 | RED | 9287.798 | -- |
| i_min | 0.004 | GREEN | -0.104 | -- |
| v_min | 0.000 | GREEN | 0.000 | -- |
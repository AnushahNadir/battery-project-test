# Section 4 — Experiments
<!-- Updated: 2026-04-16 -->
<!-- STATUS: Draft — fill in [PLACEHOLDER] values after running experiments -->

We design five experiments to answer the research question:
*Can an uncertainty-aware, auditable pipeline improve the reliability of battery RUL prediction?*

---

## Experiment 1 — Predictive Baselines

**Question:** How good are the raw predictors before reliability modules are added?

**Protocol:**
- Battery-level GroupKFold CV (k=5, groups=battery\_id) on the training set
- Final evaluation on the held-out test set
- Report RMSE and MAE per model

**Models compared:** Statistical baseline, XGBoost, TCN

**Expected outcome:** XGBoost and TCN substantially outperform the statistical baseline.
CV RMSE is the more conservative (honest) estimate; test RMSE may be optimistic
if test batteries happen to be easier than training batteries.

**See Table 2.**

---

## Experiment 2 — Uncertainty Calibration

**Question:** Are the conformal prediction intervals trustworthy?

**Protocol:**
- Calibration set: 6 held-out batteries (stratified by temperature)
- Test coverage reported per temperature group and overall
- Nominal target: 90% coverage

**Metrics reported:**
- Empirical coverage (fraction of true RUL values inside the predicted interval)
- Average interval width
- Calibration score (agreement between nominal and empirical coverage)
- Per-group breakdown: room / hot / cold

**Expected outcome:** Empirical coverage ≥ 90% overall; intervals are wider for
cold batteries (fewer training examples) and narrower for room-temperature batteries.

**See Table 3.**

---

## Experiment 3 — Reliability Stratification

**Question:** Do predictions with lower uncertainty actually have lower error?

**Protocol:**
- Divide all test observations into terciles by interval width
- Report RMSE for each tercile (narrow / medium / wide intervals)

**Expected outcome:** The narrow-interval group has the lowest RMSE.
This proves that uncertainty is practically meaningful, not just a formatting layer.

**See Figure 3.**

---

## Experiment 4 — Audit Layer Evaluation

**Question:** Does the Reliability Audit Layer separate trustworthy from risky predictions?

**Protocol:**
- Run full system on test batteries
- For each audit label (PASS / WARNING / FAIL):
  - Count proportion of batteries
  - Compute RMSE within that class
  - Compute near-term failure incidence within that class
  - Compute severe-error rate (|error| > 100 cycles) within that class

**Expected pattern:**
- PASS batteries: lowest RMSE, lowest failure incidence
- FAIL batteries: highest RMSE, highest failure incidence
- If this pattern does not hold, the audit rules are wrong and must be revised.

**Key metric: severe-error catch rate** — fraction of batteries with MAE > 100 cycles
that the audit layer assigned WARNING or FAIL. This is the paper's strongest result.

**See Table 4.**

---

## Experiment 5 — Ablation Study

**Question:** Which module actually adds reliability value?

**Configurations:**

| Config | Components |
|--------|-----------|
| A | XGBoost only |
| B | XGBoost + conformal uncertainty |
| C | XGBoost + uncertainty + survival risk |
| D | Full system (A + B + C + audit layer) |

**Metrics compared across configs:**
- RMSE / MAE
- Conformal coverage (B, C, D only)
- % predictions assigned PASS / WARNING / FAIL (D only)
- Severe-error catch rate (D only)

**Run:** `python scripts/run_ablation.py`
**Output:** `data/processed/modeling/ablation_results.csv`

**Expected outcome:** Each added module improves at least one reliability metric
without degrading the others. Config D should have the best severe-error catch rate.

**See Table 5.**

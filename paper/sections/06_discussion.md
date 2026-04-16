# Section 6 — Discussion
<!-- Updated: 2026-04-16 -->
<!-- STATUS: Draft — revise after results are finalized -->

## 6.1 Reliability Beyond RMSE

The central finding of this work is that point-estimate accuracy (RMSE) alone is
an insufficient measure of a battery RUL system's practical trustworthiness.
Our XGBoost model achieves a test RMSE of [PLACEHOLDER] cycles — well below the
100-cycle operational threshold — yet without uncertainty bounds or risk signals,
a practitioner cannot know whether to trust any given prediction.

The conformal intervals demonstrate that calibrated uncertainty is achievable
at the battery level: [PLACEHOLDER]% of true RUL values fall inside the 90%
prediction interval across all test batteries, with per-group coverage
of [PLACEHOLDER]% (room), [PLACEHOLDER]% (hot), and [PLACEHOLDER]% (cold).

## 6.2 The Audit Layer as the Core Contribution

The Reliability Audit Layer (RAL) is the paper's primary technical contribution.
Unlike a simple threshold on RMSE, the RAL integrates three orthogonal reliability
signals — uncertainty width, near-term failure probability, and anomaly detection —
into a deterministic per-battery verdict.

The key result from Experiment 4 is that the PASS class has the lowest RMSE
([PLACEHOLDER] cycles) and the FAIL class has the highest
([PLACEHOLDER] cycles), confirming that the audit labels are predictively
meaningful. The severe-error catch rate of [PLACEHOLDER]% — the fraction of
batteries with MAE > 100 cycles that were assigned WARNING or FAIL —
shows that the RAL actively prevents dangerous over-confidence.

## 6.3 Limitations

**Small test population.**
With [PLACEHOLDER] test batteries, the confidence intervals around per-battery
statistics are wide. Claims about individual battery behaviour should be interpreted
cautiously.

**CALCE generalization.**
The CALCE LOBO-CV (Experiment 1) reports a mean RMSE of [PLACEHOLDER] ± [PLACEHOLDER]
cycles across four folds. This is [PLACEHOLDER: better/worse] than the full combined
test set RMSE, suggesting that [PLACEHOLDER: interpretation].

**CV / test RMSE gap.**
The GroupKFold CV RMSE of [PLACEHOLDER] cycles is [PLACEHOLDER]× larger than the
test set RMSE of [PLACEHOLDER] cycles. This gap indicates that the held-out test
batteries may be unrepresentatively easy. The CV RMSE is the more conservative and
reliable quality estimate for new deployment scenarios.

**Survival risk as descriptive, not predictive.**
The Kaplan-Meier risk estimator is a descriptive tool for the observed test population.
It is not a generative model that transfers to unseen batteries. Future work should
replace it with a learned hazard model (e.g. Cox proportional hazards or a
neural survival model) trained on the full training population.

**RAG excluded.**
An evidence retrieval module (TF-IDF and FAISS-based) is present in the codebase
but was excluded from the core reliability evaluation. Its outputs are not tied to
measurable reliability metrics at this stage.
> "An evidence retrieval module is under active development but was excluded from
> the core reliability evaluation."

## 6.4 Broader Implications

The results support the thesis that **reliability-aware filtering matters more than
chasing lower RMSE**. A system that correctly identifies its own uncertain predictions
and flags them for human review is more deployable than a marginally more accurate
system that presents all predictions with equal confidence.

The staged architecture — prediction → uncertainty → risk → audit — is general and
transferable to other prognostic health management domains where point prediction
alone is insufficient for decision support.

# Section 3 — Methodology
<!-- Updated: 2026-04-16 -->
<!-- STATUS: Draft — fill in [PLACEHOLDER] values after running experiments -->

## 3.1 Data

We evaluate on two publicly available lithium-ion battery datasets.

**NASA PCOE Battery Dataset.**
[TODO: cite NASA PCOE] Contains discharge cycle data for 34 lithium-ion batteries
(LG 18650 cells, nominal capacity 2.0 Ah) tested under three temperature conditions:
room temperature (~25°C, batteries B0005–B0028), hot (43°C, B0029–B0040),
and cold (<10°C, B0041–B0056). Batteries B0049–B0056 exhibit anomalous early
capacity drops and are excluded from model training via a pre-training filter.

**CALCE CS2 Battery Dataset.**
[TODO: cite CALCE] Contains discharge cycle data for four batteries (CS2_35–CS2_38,
LFP cells, rated at ~1.1 Ah) tested at room temperature (~25°C) under standard
charge/discharge protocols.

**Combined dataset.**
We merge both sources into a single feature matrix after standardizing column names,
units, and EOL definitions. The combined dataset contains **38 batteries and 6,670
discharge cycles** in total.

**End-of-life definition.**
End-of-life (EOL) is defined as the first cycle at which measured capacity falls
below 80% of the battery's initial capacity (capacity\_fade\_fraction = 0.80).
Remaining Useful Life (RUL) at cycle $t$ is defined as $\text{EOL\_cycle} - t$.

**Features.**
We extract the following cycle-level features from raw time-series:
capacity (Ah), temperature (°C), voltage (V), current (A), energy (J),
discharge duration (s), and their cycle-over-cycle differences.
Full feature list: [TODO: enumerate FEATURE\_COLUMNS from ml\_model.py].

---

## 3.2 Predictive Modeling

We compare three predictors of increasing representational complexity.

**Statistical baseline.**
An exponential capacity fade model per battery:
$C(t) = C_0 \cdot e^{-\lambda t}$,
where $C_0$ is initial capacity and $\lambda$ is the per-battery degradation rate
estimated by curve fitting. RUL is derived analytically from the EOL threshold.
This model requires no cross-battery generalization and serves as the lower-bound reference.

**XGBoost (primary predictor).**
A gradient-boosted tree ensemble trained on the cycle-level feature matrix.
We train one model across all training batteries (battery-level grouped split).
Hyperparameters: n\_estimators=400, max\_depth=6, learning\_rate=0.05, subsample=0.8
(see `configs/pipeline.yaml`). The model also produces quantile predictions
($\alpha \in \{0.05, 0.50, 0.95\}$) used by the uncertainty module.

**TCN (Temporal Convolutional Network).**
A sequence model operating on windows of length 8 cycles.
Architecture: hidden\_channels=32, dropout=0.10, trained with Huber loss ($\beta=3$)
and gradient clipping (max\_norm=1.0) for [PLACEHOLDER: epochs] epochs with early stopping.
Serves as the deep-learning comparison point.

All models are trained on battery-level splits (Section 3.6). No cycles from a held-out
battery appear in any training fold.

---

## 3.3 Uncertainty Quantification

We apply split conformal prediction [TODO: cite Angelopoulos & Bates 2022] to
convert the XGBoost quantile outputs into calibrated prediction intervals.

A held-out calibration set of 6 batteries (stratified by temperature group) is used
to compute the conformal non-conformity score. We apply Leave-One-Battery-Out
calibration (LOBO) within the cold temperature group, which has fewer batteries and
requires more conservative intervals.

The target nominal coverage is 90% (conformal alpha = 0.10). We additionally apply
a safety factor of 1.20 to interval widths, making the system conservatively cautious
at the cost of slightly wider bands.

Coverage is reported per temperature group (room, hot, cold) and overall.

---

## 3.4 Near-Term Failure Risk

For each battery at each observed cycle, we estimate the probability that the battery
will reach EOL within the next **20 cycles** (configurable horizon).

We use a Kaplan-Meier survival estimator [TODO: cite] fitted on test battery
degradation trajectories. The output is a scalar failure probability per cycle,
categorised as: LOW (<0.30), MEDIUM (0.30–0.70), HIGH (>0.70).

Note: the Kaplan-Meier estimator is a descriptive tool applied to the test population.
It has no separate training phase and does not generalise to unseen batteries.
Its role in this system is to generate a near-term risk signal for the audit layer.

---

## 3.5 Reliability Audit Layer

The Reliability Audit Layer (RAL) aggregates predictions over the **last 10 observed
cycles** per battery and issues a deterministic per-battery verdict:
**PASS**, **WARNING**, or **FAIL**.

### Inputs (per battery, over observation window)

| Input | Description |
|-------|-------------|
| Uncertainty interval width | Mean width of 90% conformal interval |
| Failure probability | Maximum near-term failure probability |
| Anomaly count | Number of anomalous cycles in window |

### Decision rules

| Condition | Verdict |
|-----------|---------|
| interval\_width < 80 AND failure\_prob < 0.30 AND anomaly\_count = 0 | **PASS** |
| interval\_width ≥ 80 OR failure\_prob ≥ 0.30 OR anomaly\_count ≥ 1 | **WARNING** |
| interval\_width ≥ 120 OR failure\_prob ≥ 0.70 OR anomaly\_count ≥ 3 | **FAIL** |

FAIL takes precedence over WARNING. Thresholds were set from the 75th and 90th
percentiles of the validation set interval widths and must not be re-tuned on the test set.

### Anomaly detection

Anomalies are cycles where the absolute prediction error (|true RUL − predicted RUL|)
falls in the top 5% of the error distribution on the test set.

---

## 3.6 Evaluation Protocol

All splits are at the **battery level**. No cycles from the same battery appear in
both training and test or calibration sets. This is enforced at runtime by an
integrity check (`src/verification/check_group_split.py`) that halts the pipeline
if any overlap is detected.

**Main protocol:** Stratified battery-level split (80/20), stratified by temperature
group to preserve thermal diversity in both train and test sets.
Seed: 42.

**Cross-validation:** GroupKFold(k=5) with groups=battery\_id on the training set,
used to produce the conservative CV RMSE estimate reported alongside test RMSE.

**CALCE protocol:** Leave-one-battery-out CV across CS2\_35–CS2\_38 (4 folds),
using all NASA training batteries plus 3 CALCE batteries as the training set in each fold.

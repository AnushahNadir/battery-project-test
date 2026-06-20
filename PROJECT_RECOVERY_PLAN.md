# PROJECT_RECOVERY_PLAN.md

**Created:** 2026-06-20  
**Purpose:** Turn the current battery-RUL project into a defensible, honest, publishable paper
by reconciling every draft claim against real artifact files, removing unsupported claims,
and reframing the contribution around cross-battery RUL prediction, per-group conformal
coverage failure under distribution shift, and honest uncertainty reporting.

**Important rule:** Do not invent any number, metric, filename, or citation.
Every value in this plan is sourced from an actual file in the repository.
When a claim cannot be traced to a file, the action is DELETE.

---

# 1. Objective

The current written materials (slides, `.docx` files) describe a project that does not match
the experimental artifacts produced by the pipeline. This plan fixes that gap systematically.

The final honest story is:
- **Task:** Cross-battery RUL prediction — train on some cells, test on completely unseen cells
  under three thermal conditions. This is harder than within-battery prediction (what most
  comparison papers do) and must be explained explicitly.
- **Finding:** Split-conformal prediction coverage degrades severely under realistic
  inter-battery distribution shift. Room and hot groups achieve 100% coverage (intervals
  are over-wide). Cold group drops to **74.5%** against an 80% target. This failure is
  mechanistically explained by a RED drift report (PSI 1.0–2.6 on 5/10 features).
- **Honesty requirements:** Report XGBoost RMSE as **44.74 ± 8.38** across 5 seeds, not a
  single favorable seed. Describe a two-model ensemble (ml=0.557, dl=0.443), not three.
  Capacity ranks 7th in feature importance; cycle_index removal costs 7.1 RMSE cycles;
  capacity removal costs 0.3. This must be discussed.

---

# 2. Repository Inspection Checklist

The following files must be read and verified before any paper edits are made.
Each file is listed with what evidence it provides.

## Artifact files

| File | Path | Evidence it provides |
|------|------|---------------------|
| `model_metrics.json` | `trained_models/model_metrics.json` | XGBoost RMSE (seed 42), per-battery RMSE, DL metrics, ensemble weights, feature importance, CV results, split metadata, train/test/cal battery lists |
| `multi_seed_benchmark_5seed.json` | `trained_models/multi_seed_benchmark_5seed.json` | XGBoost mean±std across 5 seeds, DL mean±std, per-seed test batteries |
| `multi_seed_benchmark_5seed.csv` | `trained_models/multi_seed_benchmark_5seed.csv` | Tabular version of above |
| `conformal_coverage_report.json` | `data/processed/modeling/conformal_coverage_report.json` | Target coverage, overall empirical coverage, per-group coverage, q_hat per group, strategy per group |
| `uncertainty_metrics.json` | `data/processed/modeling/uncertainty_metrics.json` | Mean CI width, calibration score, risk distribution, coverage_90_percent (note: different metric from conformal) |
| `ablation_results.json` | `trained_models/ablation_results.json` | Exact feature list (10 features), RMSE with/without cycle_index, RMSE with/without capacity, delta values |
| `feature_importance.json` | `data/processed/modeling/feature_importance.json` | Ranked feature importances, directions |
| `groupkfold_cv_report.json` | `data/processed/modeling/groupkfold_cv_report.json` | 5-fold CV mean±std, per-fold RMSE, worst fold |
| `drift_report.json` | `data/processed/modeling/drift_report.json` | PSI per feature, RED/AMBER/GREEN status, alert list |
| `survival_risk_report.md` | `data/processed/modeling/survival_risk_report.md` | Event count, event rate, model type, feature weights |
| `survival_risk_metrics.json` | `data/processed/modeling/survival_risk_metrics.json` | Verify whether C-index field exists |
| `final_system_report.md` | `data/processed/modeling/final_system_report.md` | Supervisor verdict, anomaly counts, all stage results |
| `anomalies.json` | `data/processed/modeling/anomalies.json` | Whether labeled anomalies exist; structure of anomaly records |
| `uncertainty_estimates.json` | `data/processed/modeling/uncertainty_estimates.json` | Per-battery, per-cycle CI bounds; verify B0047 is absent |
| `per_cell_summary.csv` | `data/processed/per_cell_summary.csv` | Per-battery EOL threshold, RUL values, whether each reached EOL |
| `cycle_features_with_rul.csv` | `data/processed/cycle_features_with_rul.csv` | Exact RUL label values; EOL capacity threshold column to verify 70% vs 80% |
| `llm_explanations.json` | `data/processed/modeling/llm_explanations.json` | Active LLM name; verify Llama vs Gemma |
| `offline_eval_results.json` | `data/rag_eval/offline_eval_results.json` | Whether RAG retrieval metrics (Recall@3 etc.) are recorded |
| `manifest.json` | `data/processed/modeling/manifest.json` | Run timestamp, config snapshot used |
| `pipeline.yaml` | `configs/pipeline.yaml` | eol.capacity_fade_fraction, eol.nominal_capacity_ahr; conformal target; risk thresholds |
| `CLAUDE.md` | `CLAUDE.md` | --alpha description, EOL formula, active LLM model path |

## Source code files to inspect

| File | What to check |
|------|---------------|
| `src/analysis/rul.py` | How EOL threshold is computed; which alpha is read |
| `src/main.py` | Default --alpha value used in argparse |
| `src/modeling/run_full_pipeline.py` | FEATURE_COLUMNS list; ensemble exclusion logic |
| `src/modeling/ml_model.py` | FEATURE_COLUMNS definition (canonical list) |
| `src/explanation/local_rag.py` | `_LLM_BASE` active line vs commented line |
| `scripts/ablation_features.py` | Confirm CONFIGS dict used for ablation |

## Written materials to audit

| File | What to check |
|------|---------------|
| `Battery_Project_Report.docx` | Every quantitative claim; dataset name; model description; RAG description |
| `Dashboard_Report.docx` | Figures presented; battery IDs used as examples |
| `QA_Document.docx` | Q&A pairs; any numbers claimed |
| `Streamlit_Output_Explained.docx` | B0047 or other battery-specific claims |
| `paperprep/` folder | All JSON/CSV files already extracted |

---

# 3. Number-to-Artifact Reconciliation Table

**Completed:** 2026-06-20. All 20 rows verified against actual artifact files on this date.
Row 8 resolved today via direct label inspection (`eol_capacity_threshold / init_capacity` ratio = 0.800 for all 34 cells).
Row 6 resolved today via direct read of `survival_risk_metrics.json` (no c_index key present).

| # | Claim in draft/slides | Claimed value | Artifact file | Actual value in artifact | Status | Action |
|---|---|---|---|---|---|---|
| 1 | Ensemble RMSE | 18.4 cycles | `model_metrics.json` | No ensemble RMSE field. XGBoost seed-42 = **21.74**. Multi-seed XGBoost = **44.74 ± 8.38** | MISMATCH | DELETE 18.4 — appears in no artifact; report 44.74 ± 8.38 as headline |
| 2 | Conformal coverage | 91.3% at 90% target | `conformal_coverage_report.json` | Target = **0.80**; overall empirical = **91.85%** | MISMATCH | Correct to "91.85% overall at 80% target" throughout |
| 3 | Mean CI width | 47.2 cycles | `uncertainty_metrics.json` | **60.62 cycles** | MISMATCH | Correct to 60.62 everywhere |
| 4 | Three-model ensemble (0.2/0.4/0.4) | weights 0.2/0.4/0.4 | `model_metrics.json` → `uncertainty_weights` | stat=**0.0**, ml=**0.557**, dl=**0.443** | MISMATCH | Rewrite as two-model ensemble; stat baseline auto-excluded (RMSE >1.5× best) |
| 5 | 18 engineered features | 18 features | `ablation_results.json` → `full.features` | **10 features**: capacity, cycle_index, temp_mean, temp_max, v_mean, i_mean, i_min, energy_j, ah_est, duration_s | MISMATCH | Correct to 10 everywhere; list all 10 by name |
| 6 | Survival C-index | 0.76 | `survival_risk_metrics.json` ✓ CONFIRMED | Keys: n_rows, n_batteries, n_events, event_rate, horizon, model_type, notes. **No c_index key.** n_events=**5**, event_rate=**0.0078** | UNSUPPORTED | DELETE C-index; label survival "exploratory (5 observed events)" |
| 7 | Anomaly precision/recall | 0.82 / 0.79 | `anomalies.json` | Unsupervised scores only; **no ground-truth labels** in records | UNSUPPORTED | DELETE precision/recall; report "32 cycles flagged, unsupervised, no ground-truth" |
| 8 | EOL = 80% / 1.6 Ah | 80%, 1.6 Ah (fixed) | `cycle_features_with_rul.csv` ✓ CONFIRMED | Ratio = **exactly 0.800** for all 34 cells (std ≈ 0). EOL is per-cell: range **0.81–1.53 Ah**, mean **1.23 Ah**. "1.6 Ah" requires nominal 2.0 Ah init — actual cells average 1.54 Ah. | MISMATCH | Replace "1.6 Ah" with "80% of each cell's measured initial capacity (range 0.81–1.53 Ah)" |
| 9 | RAG uses Gemma 3-4B-IT | Gemma 3-4B-IT | `src/explanation/local_rag.py` + CLAUDE.md | Active model = **Llama 3.1 8B Instruct**; Gemma is commented-out alternative | MISMATCH | Replace "Gemma 3-4B-IT" with "Llama 3.1 8B Instruct" in all materials |
| 10 | Dataset = "NASA CALCE" | NASA CALCE | Battery IDs B0005–B0056 | B0005–B0056 = **NASA PCoE** (Saha & Goebel, 2007). CALCE = Univ. Maryland, different cells entirely. | MISMATCH | Replace every instance of "NASA CALCE" with "NASA PCoE" |
| 11 | RAG Recall@3 = 0.812 | 0.812 | `data/rag_eval/offline_eval_results.json` | File confirmed: **no retrieval metric**. Rubric-based eval only (hedging, mechanism, grounding, overconfidence, RUL-mention). | UNSUPPORTED | DELETE all retrieval metrics; describe as "automated rubric evaluation" |
| 12 | B0047 as test example | test battery | `model_metrics.json` → `actual_train_batteries` | B0047 confirmed in **actual_train_batteries**. Real test batteries (seed 42): B0030, B0033, B0036, B0041, B0044, B0052. | FATAL | Remove B0047 from all figures/examples; replace with B0030 (RMSE=1.93) and B0052 (RMSE=63.89) |
| 13 | Cold group coverage | (not reported in draft) | `conformal_coverage_report.json` | Cold: **74.51%** empirical vs 80% target (LOBO strategy, q_hat=31.98 cycles) | MISSING FINDING | Add per-group coverage table; this is the central finding of the uncertainty section |
| 14 | Room/hot group coverage | (not reported in draft) | `conformal_coverage_report.json` | Room: **100.0%** (q_hat=69.71, split); Hot: **100.0%** (q_hat=25.28, split) | MISSING FINDING | Add to table; note over-coverage = excessively wide intervals for room group |
| 15 | XGBoost single-seed RMSE | varies | `model_metrics.json` | Seed-42: **21.74 cycles** (MAE=10.96) | CONFIRMED | Use only with explicit "seed 42" caveat; headline must be multi-seed |
| 16 | DL (TCN/MLP) RMSE | varies | `model_metrics.json` + `multi_seed_benchmark_5seed.json` | Seed-42: **27.35**; Multi-seed (5 seeds): **41.34 ± 4.78** | CONFIRMED | Report multi-seed; DL slightly better than XGBoost in expectation |
| 17 | Per-battery RMSE spread | (not reported) | `model_metrics.json` → `per_battery_rmse` | B0030=**1.93**, B0033=**5.88**, B0036=**15.44**, B0041=**20.80**, B0044=**32.26**, B0052=**63.89** (33× spread) | MISSING | Add per-battery table to Results; cold group drives worst performance |
| 18 | Drift status | (not reported) | `drift_report.json` | **RED** overall. 5 RED features: i_mean PSI=**2.573**, temp_max=**2.462**, v_mean=**2.077**, duration_s=**1.814**, temp_mean=**1.010** | MISSING | Add drift PSI table; connect to cold-group coverage failure mechanistically |
| 19 | coverage_90_percent = conformal coverage | 96.3% | `uncertainty_metrics.json` | 96.3% is **raw quantile band coverage** ([5th, 95th] percentile predictions). Conformal coverage is **91.85%** at 80% target. Distinct metrics. | MISMATCH | Report separately with explicit labels; never substitute one for the other |
| 20 | CV RMSE | varies | `groupkfold_cv_report.json` | 5-fold CV: mean=**17.64 ± 13.73** cycles; worst fold=**43.95** cycles | CONFIRMED | Report with ± std and worst-fold value; note CV mean is optimistic vs held-out test (44.74) |

---

# 4. EOL and Label Validation Plan

**RESOLVED 2026-06-20.** Labels confirmed at **alpha=0.8** (80% of per-cell measured initial
capacity). CLAUDE.md run example fixed from `--alpha 0.7` to `--alpha 0.8`. "1.6 Ah" claim
deleted — actual per-cell thresholds range 0.81–1.53 Ah (mean 1.23 Ah). No pipeline rerun needed.

~~**The problem:** `CLAUDE.md` shows a run example using `--alpha 0.7` (EOL = 70% × initial
capacity). The config file shows `capacity_fade_fraction: 0.80` (EOL = 80% = 1.6 Ah).
The paper draft claims EOL = 80% / 1.6 Ah. These may conflict.~~

## Steps

### EOL-1: Find which alpha generated the current labels

```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/processed/cycle_features_with_rul.csv')
print('eol_capacity_threshold values (unique):')
print(df['eol_capacity_threshold'].unique())
grp = df.groupby('battery_id').first()[['init_capacity','eol_capacity_threshold']]
grp['ratio'] = grp['eol_capacity_threshold'] / grp['init_capacity']
print(grp[grp['init_capacity'] > 0.5][['init_capacity','eol_capacity_threshold','ratio']])
"
```

**Expected outcome:**
- If ratio ≈ 0.70 → labels were built at 70% EOL (alpha=0.7, EOL=1.4 Ah)
- If ratio ≈ 0.80 → labels were built at 80% EOL (config value, EOL=1.6 Ah)

### EOL-2: Check argparse default in main.py

```powershell
python -c "
src = open('src/main.py').read()
for line in src.split('\n'):
    if 'alpha' in line.lower():
        print(line)
"
```

### EOL-3: Check pipeline.yaml

```powershell
python -c "
import yaml
cfg = yaml.safe_load(open('configs/pipeline.yaml'))
print('eol section:', cfg.get('eol', {}))
"
```

### EOL-4: Decision matrix

| Scenario | Action |
|----------|--------|
| Labels at 70% AND paper claims 70% | Fix only terminology: say "70% of initial capacity" |
| Labels at 80% AND paper claims 80% | No label change needed; fix CLAUDE.md example command |
| Labels at 70% BUT paper claims 80% | **Regenerate labels** at 80%, rerun pipeline, regenerate all artifacts |
| Ambiguous | Regenerate at 80% to match config; update CLAUDE.md example |

### EOL-5: If regeneration is needed

```powershell
python -m src.main --metadata data\raw\metadata.csv --raw_root data\raw\data --out_dir outputs\run_eol_80 --alpha 0.8 --non_interactive
python -m src.modeling.run_full_pipeline
```

Then copy fresh artifacts to `paperprep/`.

### EOL-6: Consistency check after resolution

Search all `.docx`, `.md`, `.py`, `.json` files for "1.4 Ah", "1.6 Ah", "70%", "80%", "alpha"
and confirm every occurrence is consistent with the chosen EOL.

**Acceptance criteria:**
- `eol_capacity_threshold` in CSV is uniform and matches the stated EOL fraction
- `pipeline.yaml` eol section matches
- Paper draft uses one consistent EOL value with correct Ah equivalent
- CLAUDE.md example command uses the correct --alpha value

---

# 5. Train/Test Split Validation Plan

**Confirmed from artifacts:**
- Seed 42 test batteries: **B0030, B0033, B0036, B0041, B0044, B0052**
- **B0047 is in `actual_train_batteries`** — any showcase using B0047 is invalid
- Cal batteries: B0018, B0027, B0038, B0045, B0048, B0050

## Steps

### SPLIT-1: Verify test batteries from artifact (already confirmed)

Source: `trained_models/model_metrics.json` → `split_metadata`.

### SPLIT-2: Audit all written materials for B0047

```powershell
Get-ChildItem -Recurse -Include "*.md","*.txt","*.json" | Select-String "B0047"
```

Also manually search `Battery_Project_Report.docx`, `Dashboard_Report.docx`,
`QA_Document.docx`, `Streamlit_Output_Explained.docx` for "B0047".
**Any mention of B0047 as a test/prediction example must be removed.**

### SPLIT-3: Choose replacement batteries for paper showcase

Based on per-battery RMSE from `model_metrics.json`:

| Battery | Temp group | Per-battery RMSE | Role in paper |
|---------|-----------|-----------------|---------------|
| **B0030** | hot | **1.93 cycles** | Good prediction example |
| **B0033** | room | 5.88 cycles | Secondary good example |
| **B0052** | cold | **63.89 cycles** | Honest failure example |
| B0044 | cold | 32.26 cycles | Secondary failure example |

The paper **must** show both B0030 (good) and B0052 (bad). Showing only good examples
is misleading. The B0052 failure is explained by cold-group distribution shift.

### SPLIT-4: Verify uncertainty_estimates contains correct batteries

```powershell
python -c "
import json
ue = json.load(open('data/processed/modeling/uncertainty_estimates.json'))
bats = set(r['battery_id'] for r in ue)
print('Batteries in uncertainty_estimates:', sorted(bats))
print('B0047 present:', 'B0047' in bats)
print('B0052 present:', 'B0052' in bats)
"
```

**Acceptance criteria:**
- B0047 removed from all written materials
- At least B0030 (good) and B0052 (bad) shown as examples
- Both confirmed in `split_metadata.test_batteries`

---

# 6. Model Performance Validation Plan

**Already confirmed from artifacts:**

| Metric | Seed 42 | Multi-seed (5 seeds) | Source |
|--------|---------|---------------------|--------|
| XGBoost RMSE | 21.74 | **44.74 ± 8.38** | `model_metrics.json`, `multi_seed_benchmark_5seed.json` |
| DL (TCN) RMSE | 27.35 | **41.34 ± 4.78** | same |
| Statistical baseline RMSE | 45.64 | 319.66 ± 196.58 (excluded) | same |
| CV mean RMSE | 17.64 ± 13.73 | — | `groupkfold_cv_report.json` |
| CV worst fold | 43.95 | — | same |
| Ensemble weights | stat=0, ml=0.557, dl=0.443 | — | `split_metadata.uncertainty_weights` |

## Steps

### PERF-1: Per-battery RMSE table (required in paper)

Already extracted. Source: `model_metrics.json` → `per_battery_rmse`.

| Battery | Temp group | XGBoost RMSE | DL RMSE |
|---------|-----------|-------------|---------|
| B0030 | hot | 1.93 | 9.31 |
| B0033 | room | 5.88 | 7.79 |
| B0036 | room | 15.44 | 35.56 |
| B0041 | cold | 20.80 | 33.00 |
| B0044 | cold | 32.26 | 29.67 |
| B0052 | cold | 63.89 | 40.63 |

Note the pattern: room/hot batteries easy; cold batteries hard. Directly connects to cold coverage failure.

### PERF-2: Compute ensemble RMSE across seeds

The multi-seed benchmark reports XGBoost and DL separately but not the ensemble.

```powershell
python -c "
import json
ms = json.load(open('trained_models/multi_seed_benchmark_5seed.json'))
for run in ms['runs']:
    print(f'Seed {run[\"seed\"]}: XGB={run[\"xgb_rmse\"]:.2f}, DL={run[\"dl_rmse\"]:.2f}')
"
```

If per-seed ensemble predictions are not saved, report XGBoost and DL separately.
Do not fabricate an ensemble RMSE.

### PERF-3: Recommended paper performance table

| Model | RMSE seed 42 | RMSE mean ± std (5 seeds) |
|-------|-------------|--------------------------|
| Statistical baseline | 45.64 | 319.66 ± 196.58 (auto-excluded from ensemble) |
| XGBoost | 21.74 | 44.74 ± 8.38 |
| TCN (DL) | 27.35 | 41.34 ± 4.78 |
| Ensemble (ml=0.557, dl=0.443) | Compute from seed-42 predictions | Not yet computed across seeds |

### PERF-4: State honestly whether ensemble beats XGBoost

Seed 42: XGBoost RMSE=21.74, DL=27.35. XGBoost alone is better.
Multi-seed: XGBoost=44.74, DL=41.34. DL slightly better on average.
Do not claim ensemble superiority without computing ensemble RMSE per seed.

**Acceptance criteria:**
- Multi-seed RMSE reported with ± std for all models
- Ensemble described accurately as two-model (stat excluded)
- Per-battery RMSE table in paper
- No single-seed RMSE as the headline result

---

# 7. Feature and Ablation Analysis Plan

**Already confirmed from artifacts:**

10 features (from `ablation_results.json`):
`capacity, cycle_index, temp_mean, temp_max, v_mean, i_mean, i_min, energy_j, ah_est, duration_s`

Feature importance ranking (from `feature_importance.json`):

| Rank | Feature | Importance | Direction |
|------|---------|-----------|-----------|
| 1 | temp_mean | 0.276 | negative |
| 2 | i_mean | 0.196 | negative |
| 3 | cycle_index | 0.121 | negative |
| 4 | temp_max | 0.107 | negative |
| 5 | duration_s | 0.098 | negative |
| 6 | ah_est | 0.058 | negative |
| **7** | **capacity** | **0.048** | **negative** |
| 8 | i_min | 0.036 | negative |
| 9 | energy_j | 0.031 | negative |
| 10 | v_mean | 0.031 | negative |

Ablation results (from `ablation_results.json`):

| Configuration | Mean RMSE | Δ vs full |
|---------------|----------|-----------|
| Full (10 features) | 22.01 ± 8.98 | — |
| Remove cycle_index | 29.11 ± 12.20 | **+7.10 cycles** |
| Remove capacity | 22.30 ± 9.17 | **+0.29 cycles** |

## Steps

### FEAT-1: Correct feature count everywhere

Replace "18 features" with "10 features" in all written materials.
List the 10 features explicitly in the Methods section.

### FEAT-2: Verify feature columns in source

```powershell
python -c "
from src.modeling.ml_model import FEATURE_COLUMNS
print('Count:', len(FEATURE_COLUMNS))
print('Features:', list(FEATURE_COLUMNS))
"
```

### FEAT-3: Write ablation discussion (required in main text)

- `cycle_index` removal costs **+7.10 RMSE cycles** — heavy temporal dependence
- `capacity` removal costs only **+0.29 RMSE cycles** — minimal degradation signal use
- `temp_mean` is the top feature (0.276) — model primarily learns thermal context

Discussion framing: "Our ablation reveals that the XGBoost model relies primarily
on temporal position (cycle_index) and thermal context (temp_mean, i_mean) rather
than direct capacity degradation. In cross-battery prediction, the model cannot
access within-battery capacity trajectories for unseen cells and instead leverages
battery-level thermal signatures as proxies for degradation stage."

**Acceptance criteria:**
- Paper states exactly 10 features and lists them
- Ablation table in Results section
- cycle_index/capacity finding discussed in main text
- No claim of 18 features anywhere

---

# 8. Conformal Prediction and Uncertainty Validation Plan

**Already confirmed from artifacts:**

| Metric | Value | Source |
|--------|-------|--------|
| Target coverage | **0.80** | `conformal_coverage_report.json` |
| Overall empirical coverage | **91.85%** | same |
| Room empirical coverage | **100.0%** (q_hat=69.71 cycles) | same |
| Hot empirical coverage | **100.0%** (q_hat=25.28 cycles) | same |
| Cold empirical coverage | **74.51%** (q_hat=31.98 cycles) | same |
| Cold strategy | LOBO (leave-one-battery-out) | same |
| Mean CI width | **60.62 cycles** | `uncertainty_metrics.json` |
| coverage_90_percent | **96.3%** | `uncertainty_metrics.json` — raw quantile band coverage, NOT conformal |
| Calibration score | 0.837 | same |

**IMPORTANT:** `coverage_90_percent = 96.3%` is the fraction of test cycles where the
true RUL falls within the [5th, 95th] quantile predictions. This is NOT conformal coverage.
Do not conflate these two. Report them separately with clear labels.

## Steps

### CONF-1: Correct the coverage target claim everywhere

Delete every mention of "90% target" for conformal prediction. The target was **0.80 (80%)**.

### CONF-2: Per-group coverage table for paper

| Temperature group | N test cycles | Strategy | q_hat (cycles) | Empirical coverage | vs 80% target |
|------------------|--------------|---------|---------------|-------------------|---------------|
| Room | 394 | Split | 69.71 | **100.0%** | +20.0% (over-covers) |
| Hot | 40 | Split | 25.28 | **100.0%** | +20.0% (over-covers) |
| Cold | 204 | LOBO | 31.98 | **74.5%** | **−5.5% (under-covers)** |
| Overall | 638 | Mixed | — | 91.85% | +11.85% |

### CONF-3: Remove "guaranteed coverage" claims

Rewrite to: "Split-conformal prediction provides marginal coverage guarantees
under exchangeability; however, our drift analysis (PSI > 1.0 on 5/10 features)
indicates severe distribution shift between training and test batteries. Empirically,
cold-group coverage (74.5%) falls below the 80% target, consistent with theoretical
expectations under exchangeability violations."

### CONF-4: Verify PINAW

```powershell
python -c "
import json
conf = json.load(open('data/processed/modeling/conformal_coverage_report.json'))
for grp, data in conf.get('per_group', {}).items():
    print(f'{grp}: pinaw={data.get(\"pinaw\", \"NOT COMPUTED\")}')
"
```

If PINAW absent, compute from `uncertainty_estimates.json`:
`PINAW = mean(rul_upper_95 - rul_lower_5) / (RUL_max - RUL_min)`

**Acceptance criteria:**
- Coverage target stated as 80% everywhere
- Per-group coverage table in paper
- Mean CI width stated as 60.62 cycles
- "Guaranteed 90% coverage" removed
- 96.3% (quantile) and 91.85% (conformal) reported as distinct metrics
- Exchangeability limitation discussed

---

# 9. Drift Analysis Plan

**Already confirmed from `drift_report.json`:**

| Feature | PSI | Status | Train mean | Test mean | Shift |
|---------|-----|--------|-----------|----------|-------|
| i_mean | 2.573 | **RED** | 1.610 A | 1.654 A | +0.044 A |
| temp_max | 2.462 | **RED** | 34.62°C | 36.42°C | +1.81°C |
| v_mean | 2.077 | **RED** | 1.914 V | 1.365 V | −0.549 V |
| duration_s | 1.814 | **RED** | 3175 s | 2827 s | −348 s |
| temp_mean | 1.010 | **RED** | 26.91°C | 28.60°C | +1.69°C |
| capacity | 0.914 | RED* | 1.338 Ah | 1.310 Ah | −0.028 Ah |
| ah_est | 0.876 | RED* | 1.226 | 1.308 | +0.082 |
| energy_j | 0.874 | RED* | 9288 J | 8951 J | −337 J |
| i_min | 0.004 | GREEN | −0.104 A | 0.000 A | — |
| v_min | 0.000 | GREEN | 0.000 V | 0.000 V | — |

*RED threshold in this report is PSI>0.20; standard practice uses PSI>0.25 for major shift.
5 features explicitly flagged by the pipeline. All 10 features except i_min/v_min show drift.

## Steps

### DRIFT-1: Include the drift table in paper

Table goes in Results/Discussion immediately before conformal coverage discussion.
It provides the mechanistic explanation for cold-group coverage failure.

### DRIFT-2: Connect drift to coverage

Narrative: "Distribution shift analysis reveals RED-status PSI on 5/10 features
(PSI range: 1.01–2.57), far exceeding the 0.25 threshold for major shift. The
cold test group shows the highest concentration of shifted features, explaining
the 74.5% conformal coverage (below the 80% target). Room and hot groups over-cover
(100%), with the room group requiring a q_hat of 69.71 cycles — indicating
excessively wide intervals to achieve coverage on the calibration set."

### DRIFT-3: Run drift-coverage linkage script

```powershell
python scripts/drift_coverage_linkage.py
```

Output: `data/processed/modeling/drift_coverage_linkage.json`
This produces the PSI-vs-coverage scatter that becomes Figure F5.

**Acceptance criteria:**
- Drift table in paper with PSI values for 5 RED features
- Overall status stated as RED
- Connection to cold-group coverage failure explicit

---

# 10. Survival Analysis Decision Plan

**Already confirmed from `survival_risk_report.md`:**
- Model: `cox_ph_proxy_logistic_hazard` (logistic regression, NOT true Cox PH)
- Events observed: **5**
- Event rate: **0.78%**
- No C-index in any artifact

## Steps

### SURV-1: Check survival_risk_metrics.json for C-index

```powershell
python -c "
import json
from pathlib import Path
p = Path('data/processed/modeling/survival_risk_metrics.json')
if p.exists():
    data = json.loads(p.read_text())
    print(json.dumps(data, indent=2))
    print('c_index present:', 'c_index' in str(data).lower())
else:
    print('File does not exist — no C-index')
"
```

### SURV-2: Decision

| Condition | Action |
|-----------|--------|
| C-index found in artifact | Report with strong caveat: "estimated from 5 events; not reliable" |
| C-index NOT found | **DELETE the C-index claim entirely** |

**Recommended paper text:** "The pipeline includes a discrete-time hazard component
(logistic approximation to Cox proportional hazards) for near-term failure probability
estimation. With only 5 observed EOL events in the test set (event rate 0.78%),
quantitative survival metrics cannot be reliably estimated and are not reported.
The survival output is provided as an exploratory system feature."

### SURV-3: What to keep vs remove

- **Remove:** C-index 0.76, any validated survival metric
- **Keep (with caveats):** Risk tier distribution (LOW 55%, MEDIUM 23.7%, HIGH 21.3%),
  labeled "indicative only, not externally validated"
- **Keep:** Survival section in appendix as system feature demonstration

**Acceptance criteria:**
- No C-index claim in paper
- Survival section labeled exploratory
- 5-event limitation explicitly stated

---

# 11. Anomaly Detection Decision Plan

**Already confirmed:**
- `final_system_report.md`: "Anomalies detected: 32"
- No ground-truth anomaly labels exist
- Precision/recall cannot be computed

## Steps

### ANOM-1: Verify anomaly file structure

```powershell
python -c "
import json
a = json.load(open('data/processed/modeling/anomalies.json'))
print('Type:', type(a), '| Count:', len(a) if isinstance(a, list) else 'N/A')
if isinstance(a, list) and len(a) > 0:
    print('Fields:', list(a[0].keys()))
    has_labels = any('true_label' in r or 'ground_truth' in r for r in a)
    print('Has ground-truth labels:', has_labels)
"
```

### ANOM-2: Action

DELETE: precision=0.82, recall=0.79. These numbers appear in no artifact.

REPLACE WITH: "Unsupervised anomaly detection flagged 32 cycles across 6 test batteries
(approximately 5% of test cycles) based on reconstruction error exceeding the 95th
percentile training threshold. No ground-truth anomaly labels are available;
precision and recall cannot be reported."

**Acceptance criteria:**
- Precision/recall claims removed everywhere
- Anomaly count (32) reported with unsupervised qualification

---

# 12. RAG / LLM System Validation Plan

**Already confirmed from CLAUDE.md:**
- Active LLM: **Llama 3.1 8B Instruct**
- Gemma 3-4B-IT is the commented-out alternative
- RAG eval: rubric-based only (hedge/mechanism/grounding/overconfidence/RUL mention)
- No retrieval metrics in any eval artifact

## Steps

### RAG-1: Verify active LLM in source

```powershell
python -c "
src = open('src/explanation/local_rag.py').read()
for i, line in enumerate(src.split('\n')[:40], 1):
    if 'LLM_BASE' in line or 'llama' in line.lower() or 'gemma' in line.lower():
        print(f'Line {i}: {line}')
"
```

### RAG-2: Check offline eval for retrieval metrics

```powershell
python -c "
import json
from pathlib import Path
p = Path('data/rag_eval/offline_eval_results.json')
if p.exists():
    data = json.loads(p.read_text())
    print('Top-level keys:', list(data.keys()))
    print('Summary:', json.dumps(data.get('summary', {}), indent=2))
else:
    print('File missing')
"
```

### RAG-3: Paper treatment

| Claim | Evidence | Action |
|-------|---------|--------|
| Gemma 3-4B-IT | None (Llama is active) | Correct to Llama 3.1 8B Instruct |
| Recall@3 = 0.812 | None | DELETE |
| Other retrieval metrics | None | DELETE all |
| Rubric pass rate | `offline_eval_results.json` | Report if pass rate computable; label "automated rubric" |

Recommended framing: "The pipeline includes a RAG explanation module using Llama 3.1
8B Instruct (4-bit NF4 quantized, requires CUDA) with SentenceTransformer embeddings
over a domain knowledge base. Automated rubric evaluation assessed pre-computed
explanations for hedging, mechanistic reasoning, and source grounding. No retrieval
quality metrics are reported; human evaluation is left as future work."

**Acceptance criteria:**
- LLM name = Llama 3.1 8B everywhere
- Retrieval metrics removed
- RAG described as system feature with automated rubric only

---

# 13. Dataset Naming and Literature Comparison Plan

**Confirmed:** B0005–B0056 = NASA PCoE (Prognostics Center of Excellence), Saha & Goebel 2007.
CALCE = University of Maryland Center for Advanced Life Cycle Engineering — different dataset.

## Steps

### DATA-1: Remove every instance of "NASA CALCE"

```powershell
Get-ChildItem -Recurse -Include "*.md","*.txt","*.py" | Select-String "CALCE"
```

Also search `.docx` files manually. Note: doc generation scripts were already fixed
in a prior session — verify the generated outputs are also clean.

### DATA-2: Correct dataset description

"We use the NASA Prognostics Center of Excellence (PCoE) battery dataset
(Saha & Goebel, 2007), comprising 34 lithium-ion cells (B0005–B0056) cycled
under three thermal conditions: room temperature (~24°C), high temperature (43°C),
and low temperature (4°C). Four hot-group batteries (B0029–B0032) are censored
(did not reach EOL). Eight cold-group batteries (B0049–B0056) exhibit accelerated
degradation and are excluded from training but retained in test evaluation."

### DATA-3: Literature comparison table

Must include a task type column before any RMSE comparison.

| Paper | Dataset | Task type | EOL def | Reported error | Uncertainty | Multi-seed |
|-------|---------|-----------|---------|---------------|-------------|------------|
| Lin 2023 (GRU+HMM) | Oxford + NASA RW | SOH curve fitting | SOH% | RMSE 0.2–1.8% SOH | HMM residuals | No |
| Zhao 2022 (BLS-LSTM) | NASA B5/B6 + CALCE CX2 | Within-battery RUL | 70% | AE ≈ 1 cycle | None | No |
| Catelani 2021 (ESN) | NASA B0005–B0018 | Within-battery RUL | 70% | EE 0–6 cycles | t-dist bounds | No |
| **This work** | NASA PCoE (34 cells) | **Cross-battery RUL** | TBD | XGB 44.74 ± 8.38 | Split conformal per group | **Yes (5 seeds)** |

**Required note before table:** "Direct RMSE comparison is not appropriate because
the tasks differ fundamentally. Within-battery methods observe early cycles of the
target battery; cross-battery methods generalize to entirely unseen cells.
The comparison is included for completeness; task differences are the primary dimension."

**Acceptance criteria:**
- "NASA CALCE" removed from all materials
- Dataset described as NASA PCoE with correct citation
- Comparison table includes task type column
- Task difference explicitly caveated

---

# 14. Paper Reframing Plan

## New honest narrative

**Title direction:** "Cross-Battery RUL Prediction with Split-Conformal Uncertainty:
Coverage Degradation Under Inter-Battery Distribution Shift"

**One-sentence contribution:** We demonstrate that split-conformal coverage guarantees
degrade predictably under severe inter-battery distribution shift, quantify this per
temperature group, and connect the failure mechanistically to PSI-based drift detection.

## Defensible claims (keep)

1. Cross-battery RUL prediction on 34 NASA PCoE cells across three thermal groups
2. XGBoost RMSE: 44.74 ± 8.38 cycles (5 seeds)
3. DL (TCN) RMSE: 41.34 ± 4.78 cycles (5 seeds)
4. Two-model ensemble (ml=0.557, dl=0.443); statistical baseline auto-excluded
5. Cold-group conformal coverage: 74.5% (target 80%)
6. Room/hot groups: 100% coverage (over-wide intervals; q_hat=69.71 cycles for room)
7. Mean CI width: 60.62 cycles
8. Drift status: RED (PSI up to 2.57 on 5/10 features)
9. Ablation: cycle_index removal +7.10 RMSE; capacity removal +0.29 RMSE
10. temp_mean is top feature (0.276); capacity ranks 7th (0.048)
11. Per-battery RMSE: 1.93 (B0030, hot) to 63.89 (B0052, cold) — 33× spread
12. Battery-level split enforced (no cycle leakage)
13. 5-seed reproducibility reported
14. Survival: exploratory only (5 events, 0.78% rate)
15. Anomaly detection: unsupervised, 32 cycles flagged

## Claims to delete

1. RMSE = 18.4 cycles (appears in no artifact)
2. "Three-model ensemble" / weights 0.2/0.4/0.4
3. 18 features
4. Survival C-index = 0.76
5. Anomaly precision = 0.82, recall = 0.79
6. "Guaranteed 90% conformal coverage"
7. "Generalizes across thermal conditions"
8. RAG Recall@3 = 0.812 and all other retrieval metrics
9. "NASA CALCE" as dataset name
10. B0047 as any test example
11. Any single-seed RMSE as the headline result

## Section-by-section edits

| Section | Remove | Rewrite | Add |
|---------|--------|---------|-----|
| Abstract | RMSE 18.4, 90% target, 3-model, 18 features, C-index, P/R, Gemma | State cross-battery task; cite 44.74±8.38, cold 74.5%, 2-model, 10 features | Seed variability note |
| Introduction | "Outperforms SOTA", "guarantees" | Frame as harder task; gap = no per-group coverage analysis | 5-item contribution list |
| Related work | "NASA CALCE" comparisons | Position each paper with task type | Comparison table with task column |
| Methods | "18 features", "3-model (0.2/0.4/0.4)" | 10 features listed; 2-model; EOL resolved value | Battery split table |
| Results | 18.4 RMSE, B0047 example | Lead with multi-seed; add per-battery table | Ablation section; B0030/B0052 plots |
| Uncertainty | "Guaranteed 90%", CI width 47.2 | Per-group table; mean width 60.62; exchangeability caveat | Drift-coverage connection |
| Survival | C-index 0.76 | "Exploratory only, 5 events" | Event count disclosure |
| Anomaly | Precision/recall | "Unsupervised, 32 flagged, no labels" | — |
| RAG | "Gemma", retrieval metrics | "Llama 3.1 8B, automated rubric only" | CUDA requirement note |
| Limitations | (likely absent) | — | Add section with ≥5 explicit limitations |
| Conclusion | All overclaims | Coverage failure finding; seed reproducibility | — |

---

# 15. Figure and Table Replacement Plan

| # | Item | Source artifact | Script/action | Output file | Paper section | Claim supported |
|---|------|----------------|---------------|-------------|---------------|----------------|
| T1 | Reconciliation table | Section 3 of this plan | Populate manually after reading files | `paperprep/reconciliation_table.md` | Supplement | Audit trail |
| T2 | Dataset summary (34 batteries) | `per_cell_summary.csv` | Format as table | `paperprep/table_dataset.md` | Methods | Battery count, groups, censoring |
| T3 | Model RMSE across seeds | `multi_seed_benchmark_5seed.json` | Format as table | `paperprep/table_rmse_seeds.md` | Results | Multi-seed RMSE ± std |
| T4 | Per-battery RMSE | `model_metrics.json` → `per_battery_rmse` | Format as table | `paperprep/table_per_battery_rmse.md` | Results | 33× RMSE spread |
| T5 | Feature importance bar chart | `feature_importance.json` | matplotlib | `paperprep/fig_feature_importance.png` | Results | Capacity rank 7th |
| T6 | Ablation table + chart | `ablation_results.json` | Format + matplotlib | `paperprep/table_ablation.md`, `fig_ablation.png` | Results | cycle_index dominance |
| T7 | Conformal coverage by group | `conformal_coverage_report.json` | Format as table | `paperprep/table_conformal_coverage.md` | Results | Cold 74.5% finding |
| T8 | CI width by group | `uncertainty_estimates.json` | Compute per group | `paperprep/table_ci_width.md` | Results | Interval efficiency |
| T9 | Drift PSI table | `drift_report.json` | Format as table | `paperprep/table_drift_psi.md` | Discussion | RED drift status |
| T10 | Literature comparison | Related papers | Manual | `paperprep/table_literature.md` | Related work | Task difference |
| F1 | Pipeline diagram | `CLAUDE.md` | Draw/illustrate | `paperprep/fig_pipeline.png` | Methods | Architecture |
| F2 | Capacity fade (B0030 + B0052) | `cycle_features_with_rul.csv` | matplotlib | `paperprep/fig_capacity_fade.png` | Results | Data characteristics |
| F3 | RUL prediction B0030 (good) | `uncertainty_estimates.json` | matplotlib | `paperprep/fig_rul_B0030.png` | Results | Good case |
| F4 | RUL prediction B0052 (bad) | `uncertainty_estimates.json` | matplotlib | `paperprep/fig_rul_B0052.png` | Results | Honest failure case |
| F5 | PSI vs coverage scatter | `drift_coverage_linkage.json` | `scripts/drift_coverage_linkage.py` | `paperprep/fig_drift_coverage.png` | Discussion | Drift-coverage link |

---

# 16. Draft Editing Plan

## Abstract
- **Remove:** 18.4 RMSE, "90% target", "three-model", "18 features", "C-index 0.76", precision/recall, "Gemma", "NASA CALCE"
- **Add:** Cross-battery task definition; 44.74 ± 8.38 RMSE; cold 74.5% vs 80% target; two-model ensemble; 10 features; ablation finding; drift RED status
- **Cite:** `multi_seed_benchmark_5seed.json`, `conformal_coverage_report.json`, `drift_report.json`

## Introduction
- **Remove:** Overclaims about thermal generalization
- **Rewrite:** Distinguish within-battery vs cross-battery; state gap (no per-group conformal analysis)
- **Add:** Contribution list — (1) cross-battery benchmark 34 cells, (2) per-group conformal with LOBO, (3) drift-coverage linkage, (4) 5-seed reproducibility, (5) ablation of cycle_index vs capacity

## Related Work
- **Remove:** "NASA CALCE" comparisons
- **Rewrite:** Position each paper with task type; explain why direct RMSE comparison is unfair
- **Add:** Table T10 with task column

## Methods
- **Remove:** "18 features", "three-model ensemble (0.2/0.4/0.4)"
- **Rewrite:** 10 features (list them); two-model ensemble; auto-exclusion rule; EOL = resolved value; survival = logistic hazard approximation
- **Add:** Battery split table T2

## Results
- **Remove:** 18.4 RMSE, B0047 example, single-seed headline
- **Rewrite:** Lead with multi-seed RMSE; then per-battery breakdown; then conformal per group; then drift
- **Add:** Tables T3, T4, T7, T8, T9; Figures F3, F4; Ablation section with T6

## Uncertainty Section
- **Remove:** "Guaranteed 90%", CI width 47.2, "91.3% at 90% target"
- **Rewrite:** Target=80%, overall=91.85%, cold=74.5%, room/hot=100%; mean width=60.62; exchangeability violation
- **Add:** Conformal coverage table T7; note that 96.3% is quantile coverage, distinct from conformal

## Survival Section
- **Remove:** C-index 0.76
- **Rewrite:** "Exploratory; 5 events; 0.78% rate; quantitative metrics not reported"

## Anomaly Section
- **Remove:** Precision 0.82, recall 0.79
- **Rewrite:** "Unsupervised; 32 cycles; no ground-truth labels"

## RAG Section
- **Remove:** "Gemma 3-4B-IT", Recall@3 = 0.812, other retrieval metrics
- **Rewrite:** "Llama 3.1 8B Instruct; automated rubric evaluation; human evaluation is future work"

## Limitations Section (ADD — currently missing)
1. 34 batteries, one chemistry, one dataset — generalization unproven
2. Cross-battery RMSE is substantially higher than within-battery benchmarks
3. cycle_index dominates over capacity — deployment requires known cycle count
4. Survival analysis underpowered (5 observed events)
5. No human evaluation of RAG explanations
6. Conformal coverage guarantee void under distribution shift

## Conclusion
- **Remove:** "outperforms state-of-the-art", "guaranteed coverage", "generalizes across thermal conditions"
- **Keep:** "demonstrates conformal failure under distribution shift", "reproducible across 5 seeds", "provides a cross-battery benchmark"

---

# 17. Execution Order

## Phase 1 — Fatal Consistency Fixes

| ID | Task | Files | Command/action | Output | Criteria | Difficulty | Required |
|----|------|-------|----------------|--------|----------|------------|----------|
| P1-1 | Resolve EOL fraction | `cycle_features_with_rul.csv`, `configs/pipeline.yaml` | EOL-1 command | Confirmed fraction | ratio ≈ 0.70 or 0.80 consistently | Low | YES |
| P1-2 | Populate reconciliation table | All artifacts in `paperprep/` | Read each file | `reconciliation_table.md` | All 20 rows filled | Medium | YES |
| P1-3 | Confirm B0047 in training | `model_metrics.json` | Already done | Confirmed | B0047 ∈ actual_train_batteries ✓ | Done | YES |
| P1-4 | Remove "NASA CALCE" from .docx | All .docx files | Manual find+replace | Corrected .docx | Zero instances remain | Low | YES |
| P1-5 | Correct LLM name | All .docx files | Manual find+replace | Corrected .docx | Llama 3.1 8B everywhere | Low | YES |

## Phase 2 — Core Experimental Validation

| ID | Task | Files | Command/action | Output | Criteria | Difficulty | Required |
|----|------|-------|----------------|--------|----------|------------|----------|
| P2-1 | Verify 10 features | `src/modeling/ml_model.py` | FEAT-2 command | Confirmed count | FEATURE_COLUMNS has exactly 10 | Low | YES |
| P2-2 | Compute ensemble RMSE per seed | `multi_seed_benchmark_5seed.json` | PERF-2 command | Ensemble table or "not computable" | Clear statement on whether ensemble was computed | Medium | YES |
| P2-3 | Verify no C-index | `survival_risk_metrics.json` | SURV-1 command | Confirmed absent | No c_index key in file or file missing | Low | YES |
| P2-4 | Verify no anomaly labels | `anomalies.json` | ANOM-1 command | Confirmed no labels | No true_label or ground_truth field | Low | YES |
| P2-5 | Verify active LLM | `src/explanation/local_rag.py` | RAG-1 command | Confirmed Llama | Active _LLM_BASE = Llama path | Low | YES |
| P2-6 | Verify no Recall@3 | `data/rag_eval/offline_eval_results.json` | RAG-2 command | Confirmed absent | No retrieval metric keys | Low | YES |
| P2-7 | Generate drift-coverage figure | `drift_coverage_linkage.json` | `python scripts/drift_coverage_linkage.py` | F5 figure | Scatter plot with battery labels | Medium | YES |
| P2-8 | Generate B0030 + B0052 RUL plots | `uncertainty_estimates.json` | matplotlib script | F3, F4 figures | Two plots; both confirmed test batteries | Medium | YES |

## Phase 3 — Paper Reframing

| ID | Task | Files | Action | Output | Criteria | Difficulty | Required |
|----|------|-------|--------|--------|----------|------------|----------|
| P3-1 | Rewrite abstract | Current .docx | Manual | Revised abstract | All DELETE claims removed | Medium | YES |
| P3-2 | Add ablation section | `ablation_results.json` | Manual + T6 | Ablation section in paper | cycle_index/capacity finding present | Medium | YES |
| P3-3 | Add per-group coverage table | `conformal_coverage_report.json` | Manual + T7 | Table in paper | cold 74.5%, room/hot 100% shown | Low | YES |
| P3-4 | Add drift table | `drift_report.json` | Manual + T9 | Table in paper | 5 RED features with PSI values | Low | YES |
| P3-5 | Rewrite conformal claim | .docx files | Manual | Revised uncertainty section | "Guaranteed 90%" removed | Medium | YES |
| P3-6 | Add limitations section | New | Manual | Limitations section | ≥ 5 explicit limitations | Low | YES |
| P3-7 | Fix literature comparison | Related papers | Manual + T10 | Comparison table | Task type column present | Medium | YES |
| P3-8 | Replace B0047 examples | .docx files | Manual | B0030 + B0052 shown | Both confirmed test batteries | Medium | YES |

## Phase 4 — Optional Strengthening

| ID | Task | Action | Output | Difficulty | Required | Expected value |
|----|------|--------|--------|------------|----------|----------------|
| P4-1 | Human RAG evaluation (n ≥ 20) | Two raters; 4-point rubric | `human_eval_results.json` | High | No | Allows reporting human eval |
| P4-2 | Second dataset validation | Download CALCE or NASA RW; run pipeline | Second-dataset RMSE table | High | No | Greatly strengthens generalization claim |
| P4-3 | Extended ablation (cycle_index-free model) | Add config to ablation script | Updated ablation table | Low | No | Directly addresses reviewer concern on leakage |
| P4-4 | Per-group PINAW computation | Compute from uncertainty_estimates.json | PINAW per group | Low | No | Interval efficiency metric |

---

# 18. Acceptance Criteria for Final Project

The project is ready to submit only when ALL of the following are true:

- [ ] **EOL consistency:** `eol_capacity_threshold` in CSV matches `pipeline.yaml`; paper states one consistent value with correct Ah equivalent
- [ ] **No B0047:** B0047 does not appear as a test example in any written material
- [ ] **Test batteries confirmed:** Paper shows B0030 (good) and B0052 (bad); both in `split_metadata.test_batteries`
- [ ] **RMSE with ± std:** XGBoost = 44.74 ± 8.38 (5 seeds); no single-seed number as headline
- [ ] **Two-model ensemble:** Paper states ml=0.557, dl=0.443; statistical baseline auto-excluded
- [ ] **10 features:** Stated and listed by name
- [ ] **Ablation in main text:** +7.10 (cycle_index), +0.29 (capacity) with discussion
- [ ] **Conformal target = 80%:** No "90% target" anywhere
- [ ] **Per-group coverage table:** Cold=74.5%, room=100%, hot=100%
- [ ] **Mean CI width = 60.62 cycles:** Stated correctly
- [ ] **No "guaranteed coverage":** Exchangeability limitation acknowledged
- [ ] **Drift table:** 5 RED features with PSI values; connected to coverage failure
- [ ] **No C-index:** Survival labeled exploratory; 5 events stated
- [ ] **No anomaly P/R:** Unsupervised count only
- [ ] **Llama 3.1 8B:** RAG model correctly named everywhere
- [ ] **No RAG retrieval metrics:** Recall@3 and similar removed
- [ ] **"NASA PCoE" everywhere:** Zero instances of "NASA CALCE"
- [ ] **Task difference stated:** Literature comparison caveats within-battery vs cross-battery
- [ ] **Limitations section:** ≥ 5 explicit limitations present
- [ ] **Every number traceable:** Each claim maps to an artifact file

---

# 19. Do Not Execute Destructive Changes Yet

**This plan is read-only. Prohibited until Phase 1 is complete:**

- Do NOT delete or overwrite any `.docx` files
- Do NOT delete any existing JSON or CSV artifact files
- Do NOT regenerate RUL labels until EOL fraction is confirmed (Task P1-1)
- Do NOT rerun the full modeling pipeline
- Do NOT edit any manuscript section
- Do NOT commit any changes

---

# Summary

## What this plan does

Converts 30 reviewer criticisms into executable tasks grounded in real artifact values
read from the repository on 2026-06-20. Every number in this plan comes from an actual file.
Fatal errors (wrong metrics, wrong battery ID, wrong model description) are fixed before
framing work begins.

## The honest paper this project can become

Not: "We built an accurate RUL predictor."  
But: "We conducted a reproducible cross-battery RUL benchmark and showed that
split-conformal coverage degrades predictably under severe inter-battery distribution
shift — quantified per temperature group and connected mechanistically to PSI drift detection."

## First 5 commands to run next

**Action 1 — Resolve EOL fraction (30 min)**
```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/processed/cycle_features_with_rul.csv')
grp = df.groupby('battery_id').first()[['init_capacity','eol_capacity_threshold']]
grp['ratio'] = grp['eol_capacity_threshold'] / grp['init_capacity']
print(grp[grp['init_capacity'] > 0.5])
"
```

**Action 2 — Confirm B0047 is training battery (5 min)**
```powershell
python -c "
import json
mm = json.load(open('trained_models/model_metrics.json'))
sm = mm['split_metadata']
print('TEST:', sm['test_batteries'])
print('B0047 in train:', 'B0047' in sm['actual_train_batteries'])
"
```

**Action 3 — Check for C-index in survival artifacts (5 min)**
```powershell
python -c "
import json
from pathlib import Path
p = Path('data/processed/modeling/survival_risk_metrics.json')
print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else 'File missing — no C-index')
"
```

**Action 4 — Verify feature count (5 min)**
```powershell
python -c "
from src.modeling.ml_model import FEATURE_COLUMNS
print(len(FEATURE_COLUMNS), list(FEATURE_COLUMNS))
"
```

**Action 5 — Search for NASA CALCE in text files (5 min)**
```powershell
Get-ChildItem -Recurse -Include "*.md","*.txt","*.py","*.json" | Select-String "CALCE" | Select-Object Path, LineNumber, Line
```

After these 5 actions, proceed to Phase 2 (P2-1 through P2-8) in order.

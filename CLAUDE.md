# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# PyTorch with CUDA (optional, falls back to sequence-MLP):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Stage 1–2: Preprocessing**
```powershell
python -m src.main --metadata data\raw\metadata.csv --raw_root data\raw\data --out_dir outputs\run_001 --alpha 0.7 --non_interactive
```

**Stage 3–6: Modeling pipeline**
```powershell
python -m src.modeling.run_full_pipeline
```

**Dashboard**
```powershell
streamlit run dashboard/app.py
```

**Tests**
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
# Single test file:
python -m unittest tests.test_invariants -v
```

**Notebook validation**
```powershell
python scripts\check_notebook.py battery_analysis.ipynb
```

**Release snapshot**
```powershell
python scripts\make_snapshot.py --tag release_candidate
```

**Override config path**
```powershell
$env:BATTERY_CONFIG = "configs/pipeline.yaml"
```

## Architecture

This is an end-to-end NASA lithium-ion battery RUL (Remaining Useful Life) prediction pipeline with two sequential stages:

### Stage 1–2: Preprocessing (`src/main.py` → `src/pipeline/run_analysis.py`)
Ingests raw metadata CSV + per-cycle time-series CSVs. Flow:
1. Load + standardize metadata columns via `src/pipeline/mapper.py` (handles fuzzy/interactive column mapping)
2. Build discharge cycle table and compute RUL from capacity fade (`src/analysis/rul.py`)
3. Gating check — validates one representative time-series file and computes a `confidence` score (`src/pipeline/confidence.py`). Rejects dataset if confidence < `gating.low_threshold` (0.60). If below `review_threshold` (0.85), writes `trigger_rag_assist.txt = True` to flag that RAG assistance should be invoked.
4. Extract time-series engineered features for all cycles (`src/analysis/degradation_features.py`)
5. Merge into `data/processed/cycle_features_with_rul.csv` — the primary input for Stage 3+

### The `--alpha` parameter
`--alpha` is the EOL capacity-fade fraction passed to `src/analysis/rul.py`. It sets `EOL_capacity = alpha × initial_capacity`. At `--alpha 0.7`, EOL = 1.4 Ah (70% of nominal 2.0 Ah). **This is not a quality or confidence threshold** — it controls when a battery is considered "dead" for RUL label generation. The gating quality thresholds (0.60 hard-reject, 0.85 RAG-trigger) are entirely separate and live in `configs/pipeline.yaml`. Using `--alpha 0.7` with the default gating config means the dataset passes the hard-reject gate but sits between the two thresholds, so RAG assistance is triggered.

### Stage 3–6: Modeling (`src/modeling/run_full_pipeline.py`)
Reads `cycle_features_with_rul.csv`. Batteries are split by temperature group (cold/hot/room) to prevent leakage. A subset of train batteries is held out as a conformal calibration set.

- **Stage 3**: Trains three models: statistical baseline (`src/modeling/statistical_baseline.py`), XGBoost (`src/modeling/ml_model.py`), and a TCN/MLP sequence model (`src/modeling/dl_sequence_model.py`). Ensemble weights are computed via inverse-RMSE weighting (see below).
- **Stage 4**: Uncertainty quantification (`src/modeling/uncertainty.py`) + conformal prediction intervals (`src/modeling/conformal.py`) with adaptive per-temperature-group calibration. Survival/hazard risk (`src/analysis/survival_risk.py`). PSI-based drift monitoring (`src/pipeline/drift.py`).
- **Stage 4.5**: Anomaly detection on test set (`src/modeling/anomaly_detection.py`).
- **Stage 5**: SHAP-based reasoning, counterfactual generation, degradation hypotheses (`src/modeling/reasoning.py`).
- **Stage 6**: Supervisor review — checks RMSE thresholds, CV ratio, hallucination indicators, writes `final_system_report.md` (`src/modeling/supervisor_review.py`).

### Ensemble weights (inverse-RMSE)
After training, each model's test RMSE is computed. Weights are assigned proportional to `1/RMSE`. Any model with RMSE > 1.5× the best model's RMSE is excluded from the ensemble entirely. The ML (XGBoost) model is always guaranteed to contribute even if it underperforms. The resulting `stat`/`ml`/`dl` weights are logged and saved in `trained_models/model_metrics.json` under `uncertainty_weights`.

### Survival / hazard risk (`src/analysis/survival_risk.py`)
The config key `survival.method` accepts `kaplan_meier` or `cox_ph`, but **`cox_ph` is not a true Cox proportional-hazards model** — it is approximated with a discrete-time logistic hazard (sklearn `LogisticRegression`, L2 regularized). The output file `survival_risk_model.json` records `model_type: "cox_ph_proxy_logistic_hazard"` so this is explicit in artifacts. With `kaplan_meier`, grouped KM curves are fitted per temperature group. The active config sets `method: cox_ph`.

### Configuration (`configs/pipeline.yaml` / `src/config.py`)
All thresholds and hyperparameters live in `configs/pipeline.yaml`. Never hardcode magic numbers in source files. `get_config()` is `lru_cache`-wrapped — call `reload_config()` if you change the YAML mid-process. Config is validated with Pydantic; cross-field invariants (e.g., `risk.horizon_cycles == survival.horizon_cycles`) are enforced at load time.

### RAG Explanation Module (`src/explanation/local_rag.py`)
Real local LLM pipeline: SentenceTransformer embeddings → ChromaDB vector store → Llama 3.1 8B generation (4-bit NF4 quantized via bitsandbytes). **Requires CUDA** — raises `RuntimeError` at generation time if no GPU is found.

**Model weights to download from HuggingFace (not included in repo):**
```
models/llama-3.1-8b-instruct/   ← active LLM (meta-llama/Llama-3.1-8B-Instruct)
models/all-MiniLM-L6-v2/        ← embeddings (sentence-transformers/all-MiniLM-L6-v2)
# Alternative (smaller, swap in local_rag.py):
models/gemma-3-4b-it/           ← google/gemma-3-4b-it
```

To switch to Gemma, swap the commented `_LLM_BASE` lines in `src/explanation/local_rag.py:29-30`.

**Knowledge base sources indexed by ChromaDB:**
- `data/knowledge_base/*.txt` — curated domain text (checked in)
- `data/papers/*.pdf` — research papers (add your own, then call `rag.reindex()`)

The ChromaDB index is persisted at `data/vector_db/`. First run processes all documents (slow); subsequent runs load the saved index instantly.

### Agent layer (`src/agents/`)
Multi-agent validation: `schema_inspector.py`, `semantic_mapper.py`, `validation_gating.py`, `supervisor_critic.py`, `run_hostile_validation.py`. The supervisor critic explicitly checks for hallucinated column mappings and rejects runs that trigger hallucination indicators.

### Dashboard (`dashboard/app.py`)
Streamlit app. Reads all artifacts from `data/processed/modeling/` and `trained_models/` at startup. Run Stage 3–6 first to populate artifacts before launching.

## Key data flow

```
data/raw/metadata.csv
data/raw/data/*.csv (per-cycle time-series)
        ↓  Stage 1-2
data/processed/cycle_features_with_rul.csv
        ↓  Stage 3-6
trained_models/          ← model artifacts + calibrators + drift reference
data/processed/modeling/ ← uncertainty, anomalies, hypotheses, counterfactuals, report
        ↓
dashboard/app.py         ← reads all of the above
```

## Important constraints

- **Battery-level train/test split** is mandatory. Never split by row/cycle — all cycles from one battery must stay in the same split to prevent leakage. This is enforced via `GroupKFold` in CV and explicit battery-list checks at pipeline start.
- **EOL threshold**: End-of-life is defined as capacity falling below `eol.capacity_fade_fraction * eol.nominal_capacity_ahr` (default: 0.80 × 2.0 = 1.6 Ah). Changing this changes all RUL labels.
- **Conformal calibration batteries** are carved out from the training set, not from test. The pipeline hard-exits if any overlap is detected between train/cal/test sets.
- **DL backend**: `force_backend: auto` uses PyTorch TCN if available, otherwise falls back to a sequence-MLP silently. Set `force_backend: torch_tcn` or `keras` to force a specific backend.
- **Flagged batteries** (`anomaly.flagged_battery_prefixes`): rows from these batteries with capacity below `capacity_floor_factor * median` are removed from training only — not from calibration or test.

## Troubleshooting

**PyTorch not installed / no CUDA for DL model**: Stage 3 DL training falls back to a sequence-MLP automatically — no action needed. RAG generation (`BatteryRAG.explain()`) is different: it hard-requires CUDA and raises `RuntimeError` with pip install instructions if unavailable.

**Pipeline hard-exits with "Battery leakage detected" or "Calibration/train overlap"**: This means the battery split produced overlapping sets, usually because there are too few batteries for the configured `n_cal_batteries`. Reduce `split.n_cal_batteries` in `pipeline.yaml` or add more battery data.

**Pipeline hard-exits with "Need at least 2 batteries"**: The preprocessed CSV has only one unique `battery_id`. Check that the metadata file has multiple batteries and that the `battery_id` column was correctly mapped.

**ChromaDB collection empty after first run**: If `data/knowledge_base/` and `data/papers/` are both empty, the index builds with zero chunks. Add `.txt` files to `data/knowledge_base/` or PDFs to `data/papers/`, then call `rag.reindex()`.

**Stage 3–6 fails with "Preprocessed dataset not found"**: Run Stage 1–2 first to generate `data/processed/cycle_features_with_rul.csv`.

**Dashboard shows no data**: All panels read from `data/processed/modeling/`. Run the full modeling pipeline (`python -m src.modeling.run_full_pipeline`) before launching Streamlit.

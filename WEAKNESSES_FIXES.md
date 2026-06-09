# Weakness Fixes

Three concrete issues identified in the current codebase, with exact file locations and the code changes needed to fix each one.

---

## Fix 1 — `pattern_agreement_ratio` is a hardcoded boolean

**File:** `src/pipeline/run_analysis.py:95`

**Problem:** The 0.25-weighted component of the gating confidence score is not measured — it is hardcoded to `0.75` if schema passes, `0.40` if it fails. This makes the confidence score partially fake.

**Current code:**
```python
# simple placeholder: if schema_ok assume decent agreement
pattern_agreement_ratio = 0.75 if validation.schema_ok else 0.40
```

**Fix:** Compute it from the actual data. `validate_timeseries()` already writes per-column min/max into `validation.metrics`. Use those to calculate what fraction of each column's value range overlaps with the expected physical bounds from `configs/pipeline.yaml`.

**Replace the placeholder with:**
```python
from src.config import get_config
from src.pipeline.schema import get_schema_ranges

def _compute_pattern_agreement(df_ts: pd.DataFrame, validation_metrics: dict) -> float:
    """
    Fraction of TS columns whose observed value range meaningfully overlaps
    with the expected physical range from configs/pipeline.yaml.
    """
    cfg = get_config()
    s = cfg.schema
    col_bounds = {
        "voltage_measured":     (s.voltage_v.min,     s.voltage_v.max),
        "voltage_load":         (s.voltage_v.min,     s.voltage_v.max),
        "current_measured":     (s.current_a.min,     s.current_a.max),
        "current_load":         (s.current_a.min,     s.current_a.max),
        "temperature_measured": (s.temperature_c.min, s.temperature_c.max),
    }
    scores = []
    for col, (lo, hi) in col_bounds.items():
        if col not in df_ts.columns:
            continue
        x = pd.to_numeric(df_ts[col], errors="coerce").dropna().to_numpy()
        if len(x) == 0:
            scores.append(0.0)
            continue
        # Fraction of values within the expected physical window
        in_range = float(np.mean((np.abs(x) >= lo) & (np.abs(x) <= hi)))
        scores.append(in_range)
    return float(np.mean(scores)) if scores else 0.0

pattern_agreement_ratio = _compute_pattern_agreement(df_ts_std, validation.metrics)
```

**Why this matters:** With the fix, a dataset that maps columns correctly but has voltages consistently reading 5 V (above the 4.2 V max) will score lower than one that passes physics checks cleanly. The gating decision becomes honest.

---

## Fix 2 — `cox_ph` config value runs logistic regression

**Files:** `src/analysis/survival_risk.py:460–462`, `src/config.py:204`, `configs/pipeline.yaml:94`

**Problem:** `survival.method: cox_ph` is a valid config value that users would expect to run a Cox Proportional-Hazards model. Instead it runs `sklearn LogisticRegression`. The artifact correctly labels itself `cox_ph_proxy_logistic_hazard`, but the config name is still misleading.

**Two-part fix:**

**Part A — Add `lifelines` and implement real Cox PH** (preferred)

Add to `requirements.txt`:
```
lifelines>=0.27
```

In `src/analysis/survival_risk.py`, replace the `else` branch (line ~453) with:
```python
    elif method == "cox_ph":
        from lifelines import CoxPHFitter

        # Build battery-level survival table required by lifelines
        batt_table = (
            pp[["battery_id", "event_cycle_per_battery", "event_observed_per_battery", "temp_group"] + feats]
            .drop_duplicates(subset=["battery_id"])
            .copy()
        )
        batt_table = batt_table.rename(columns={
            "event_cycle_per_battery": "duration",
            "event_observed_per_battery": "event",
        })
        batt_table["duration"] = pd.to_numeric(batt_table["duration"], errors="coerce").fillna(1).clip(lower=1)
        for f in feats:
            batt_table[f] = pd.to_numeric(batt_table[f], errors="coerce").fillna(0)

        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(batt_table[["duration", "event"] + feats], duration_col="duration", event_col="event")
        except Exception as e:
            notes.append(f"CoxPH fit failed ({e}); falling back to logistic hazard.")
            # fall through to logistic below
        else:
            pp_std, medians, stds = _impute_and_standardize(pp, feats)
            pp_std["hazard_prob"] = np.clip(
                cph.predict_partial_hazard(pp_std[feats].fillna(0)).to_numpy(dtype=float), 0.0, 1.0
            )
            pp_std["failure_prob_horizon"] = 0.0
            for bid, g in pp_std.groupby("battery_id", sort=False):
                g_sorted = g.sort_values("cycle_index", kind="mergesort")
                fp = _failure_prob_within_horizon(g_sorted["hazard_prob"].values, horizon=horizon)
                pp_std.loc[g_sorted.index, "failure_prob_horizon"] = fp
            pp_std["risk_category"] = pp_std["failure_prob_horizon"].apply(
                lambda p: cfg.risk.risk_category(float(p))
            )
            preds = pp_std
            model_type = "cox_proportional_hazards"
            model_blob = {
                "model_type": model_type,
                "method": method,
                "feature_cols": feats,
                "coef": cph.params_.to_dict(),
                "horizon": horizon,
                "notes": notes,
            }
```

**Part B — If you don't want `lifelines`**, at minimum rename the config value so it is honest:

In `src/config.py:204`, update the validator:
```python
allowed = {"kaplan_meier", "discrete_hazard"}   # remove "cox_ph"
```

In `configs/pipeline.yaml:94`:
```yaml
survival:
  horizon_cycles: 30
  method: discrete_hazard   # was: cox_ph
```

And in `src/analysis/survival_risk.py:460`, remove the misleading note since the name now matches:
```python
# before:
if method == "cox_ph":
    notes.append("Configured method='cox_ph' is approximated with logistic discrete-time hazard in this version.")
    model_type = "cox_ph_proxy_logistic_hazard"

# after (Part B only — no lifelines):
model_type = "discrete_time_logistic_hazard"
```

---

## Fix 3 — RAG fails silently on CPU / no-GPU machines

**File:** `src/explanation/local_rag.py:262–276`

**Problem:** Retrieval (ChromaDB + SentenceTransformer) works on CPU, but generation hard-raises `RuntimeError` the moment `explain()` is called without a GPU. There is no fallback — callers get an exception with no useful output.

**Fix:** Add a `retrieval_only` fallback mode. When CUDA is unavailable, `explain()` returns the top-k retrieved chunks directly instead of generating an LLM response. Retrieval is useful on its own — it surfaces the most relevant literature passages for the query.

**Changes to `BatteryRAG.__init__`:**
```python
def __init__(
    self,
    project_root: Path,
    llm_path: str = _LLM_BASE,
    embedding_path: str = _MINILM_BASE,
    pdf_folders: Optional[List[str]] = None,
    generation_backend: str = "auto",   # "auto" | "local_llm" | "retrieval_only"
):
    ...
    self._generation_backend = generation_backend
    self._init_db()
```

**Replace `_load_llm` with a guarded version:**
```python
def _load_llm(self) -> bool:
    """Returns True if LLM loaded, False if falling back to retrieval-only."""
    if self._llm_loaded:
        return True

    import torch
    if not torch.cuda.is_available():
        if self._generation_backend == "local_llm":
            raise RuntimeError(
                "generation_backend='local_llm' requires CUDA. "
                "Use generation_backend='retrieval_only' for CPU-only environments."
            )
        logger.warning(
            "CUDA not available — RAG running in retrieval-only mode. "
            "Responses will be retrieved passages, not LLM-synthesised answers."
        )
        return False

    # ... existing BitsAndBytesConfig + model load unchanged ...
    self._llm_loaded = True
    return True
```

**Replace the `explain()` method body:**
```python
def explain(self, query: str, extra_context: str = "", top_k: int = 5) -> Tuple[str, List[str]]:
    if self._collection is None or self._collection.count() == 0:
        return "Knowledge base is empty. Run reindex() to build the index.", []

    query_embedding = self._embedder.encode([query]).tolist()
    results = self._collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, self._collection.count()),
        include=["documents", "metadatas"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = sorted({m.get("source", "") for m in metadatas})

    llm_available = self._load_llm()

    if not llm_available or self._generation_backend == "retrieval_only":
        # Retrieval-only: return chunks directly, clearly labelled
        passages = "\n\n---\n\n".join(
            f"[{m.get('source', 'unknown')}]\n{doc}"
            for doc, m in zip(documents, metadatas)
        )
        answer = f"(Retrieval-only mode — no LLM generation)\n\n{passages}"
        return answer, sources

    prompt = self._format_prompt(query, documents, metadatas, extra_context)
    answer = self.generate_answer(prompt)
    return answer, sources
```

**Usage after fix:**
```python
# Full LLM mode (GPU required — existing behaviour)
rag = BatteryRAG(project_root=BASE)

# Explicit retrieval-only (CPU safe, no model weights needed)
rag = BatteryRAG(project_root=BASE, generation_backend="retrieval_only")

# Auto-detect (GPU → LLM, no GPU → retrieval-only, no crash)
rag = BatteryRAG(project_root=BASE, generation_backend="auto")  # default
```

No changes to `requirements.txt` needed for this fix — `sentence-transformers` and `chromadb` already run on CPU.

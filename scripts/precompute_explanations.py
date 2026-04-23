"""
Pre-compute LLM explanations for all configured test batteries.

Saves results to:
    data/processed/modeling/llm_explanations.json

Run from the battery-rul project root:
    python scripts/precompute_explanations.py

After running, restart the Streamlit dashboard — it will load these results
instantly instead of calling the LLM at request time.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# ── CONFIGURE: which batteries to process ─────────────────────────────────────
# Comment/uncomment to control which batteries are pre-computed.
# Full test set: B0030, B0033, B0036, B0041, B0044, B0052

BATTERIES_TO_COMPUTE = [
    "B0036",
    "B0041",
    "B0030",
    "B0033",
    "B0044",
    "B0052",
]

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
MDL  = BASE / "data" / "processed" / "modeling"
TM   = BASE / "trained_models"
OUT  = MDL / "llm_explanations.json"

sys.path.insert(0, str(BASE))

import pandas as pd
from src.explanation.explanation_builder import ExplanationBuilder


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_context(battery_id, df, u_df, a_df, survival, metrics,
                  feat_imp, hypotheses, counterfact, drift, unc_metrics,
                  RUL_COL, has_bands):
    """Replicate the dashboard context-building logic for one battery."""
    df_b = df[df["battery_id"].astype(str) == battery_id].sort_values("cycle_index")
    u_b  = u_df[u_df["battery_id"].astype(str) == battery_id].sort_values("cycle_index") \
           if RUL_COL and not u_df.empty and "battery_id" in u_df.columns else pd.DataFrame()
    a_b  = a_df[a_df["battery_id"].astype(str) == battery_id] \
           if not a_df.empty and "battery_id" in a_df.columns else pd.DataFrame()
    s_b  = survival[survival["battery_id"].astype(str) == battery_id].sort_values("cycle_index") \
           if not survival.empty and "battery_id" in survival.columns else pd.DataFrame()

    latest_rul = None
    if not u_b.empty and RUL_COL:
        latest_rul = round(float(u_b.iloc[-1][RUL_COL]))

    ctx = []

    # ── 1. Latest cycle measurements ──────────────────────────────────────────
    if not df_b.empty:
        last = df_b.iloc[-1]
        cap   = last["capacity"]
        eol_t = last["eol_capacity_threshold"]
        init  = last["init_capacity"]

        if pd.isna(cap):
            cap_str = "unavailable (NaN — measurement missing, do NOT interpret as a capacity value)"
            eol_cmp = "cannot be determined (capacity is missing)"
            fade_str = "unknown"
        else:
            cap_str = f"{cap:.4f} Ah"
            if cap < eol_t:
                eol_cmp = (f"BELOW EOL threshold ({cap:.4f} < {eol_t:.4f})"
                           f" — battery HAS crossed end-of-life")
            else:
                margin = cap - eol_t
                eol_cmp = (f"ABOVE EOL threshold ({cap:.4f} > {eol_t:.4f})"
                           f" — margin remaining: {margin:.4f} Ah")
            if not pd.isna(init) and init > 0:
                abs_drop = init - cap
                fade_str = f"{abs_drop:.4f} Ah lost ({abs_drop / init * 100:.1f}% fade)"
            else:
                fade_str = "unknown"

        ctx.append(
            f"LATEST CYCLE ({int(last['cycle_index'])}):\n"
            f"  Capacity: {cap_str} (initial: {init:.4f} Ah — capacity drop: {fade_str})\n"
            f"  EOL threshold: {eol_t:.4f} Ah — capacity is {eol_cmp}\n"
            f"  Voltage mean/min: {last['v_mean']:.3f} / {last['v_min']:.3f} V\n"
            f"  Current min: {last['i_min']:.4f} A\n"
            f"  True RUL at this cycle: {int(last['RUL'])} cycles"
        )

    # ── 2. RUL prediction + uncertainty (per-battery) ─────────────────────────
    if latest_rul is not None and not u_b.empty:
        lo = float(u_b["rul_lower_5"].iloc[-1]) if has_bands else None
        hi = float(u_b["rul_upper_95"].iloc[-1]) if has_bands else None
        if lo is not None and hi is not None:
            width = hi - lo
            interval = (f" | 90% CI for THIS battery: [{lo:.0f}, {hi:.0f}]"
                        f" (CI width = {width:.0f} cycles)")
        else:
            interval = ""

        rul_notes = []
        if latest_rul <= 0:
            rul_notes.append(
                "MANDATORY STATEMENT: RUL <= 0 — this battery has already "
                "reached or passed its end-of-life threshold. "
                "You MUST state this explicitly. "
                "Do NOT reframe the CI upper bound as remaining useful life — "
                "the CI upper bound represents model uncertainty, NOT additional life."
            )
        if lo is not None and lo <= 0:
            rul_notes.append(
                f"CI lower bound = {lo:.0f}: the model cannot rule out that "
                "this battery is already at or past EOL."
            )
        rul_note = (" | " + " | ".join(rul_notes)) if rul_notes else ""
        ctx.append(f"RUL PREDICTION: {latest_rul} cycles{interval}{rul_note}")

    # ── 3. Failure risk ────────────────────────────────────────────────────────
    if not s_b.empty and "failure_prob_horizon" in s_b.columns:
        last_s = s_b.iloc[-1]
        fp = float(last_s["failure_prob_horizon"])
        rc = str(last_s.get("risk_category", "UNKNOWN"))
        ctx.append(f"FAILURE RISK (next 20 cycles): {fp*100:.1f}% — {rc}")

    # ── 4. Anomalies ───────────────────────────────────────────────────────────
    if not a_b.empty:
        types = (", ".join(sorted(a_b["anomaly_type"].unique()))
                 if "anomaly_type" in a_b.columns else "unknown")
        scores = a_b["anomaly_score"].astype(float) if "anomaly_score" in a_b.columns else None
        worst  = f" | worst score: {scores.max():.1f}" if scores is not None else ""
        ctx.append(f"ANOMALIES: {len(a_b)} anomalous cycles — types: {types}{worst}")
    else:
        ctx.append("ANOMALIES: none detected")

    # ── 5. Top feature importances ─────────────────────────────────────────────
    if feat_imp:
        top5 = feat_imp[:5]
        fi_lines = "  |  ".join(
            f"{f['feature']} {f['importance']*100:.1f}% ({f.get('direction','')})"
            for f in top5
        )
        ctx.append(f"TOP PREDICTIVE FEATURES: {fi_lines}")

    # ── 6. Degradation hypotheses ──────────────────────────────────────────────
    if hypotheses:
        hyp_lines = "\n".join(
            f"  - {h['hypothesis_text']} [confidence: {h.get('confidence', 0):.2f}]"
            for h in hypotheses[:4]
        )
        ctx.append(f"DEGRADATION HYPOTHESES (model-derived):\n{hyp_lines}")

    # ── 7. Counterfactuals ─────────────────────────────────────────────────────
    bat_cf = [c for c in counterfact
              if str(c.get("observation_id", "")).startswith(battery_id)]
    if bat_cf:
        cf_lines = "\n".join(
            f"  - If {c['counterfactual']['feature_changed']} changed "
            f"from {c['counterfactual']['original_value']:.3f} to "
            f"{c['counterfactual']['counterfactual_value']:.3f} → "
            f"RUL would change by {c['counterfactual']['predicted_rul_change']:+.1f} cycles"
            for c in bat_cf[:3]
        )
        ctx.append(f"COUNTERFACTUAL WHAT-IFS:\n{cf_lines}")

    # ── 8. Drift alerts ────────────────────────────────────────────────────────
    if drift:
        alerts = drift.get("alerts", [])
        status = drift.get("overall_status", "unknown")
        ctx.append(
            f"DATA DRIFT: overall status = {status}"
            + (f" | alerts: {', '.join(alerts)}" if alerts else "")
        )

    # ── 9. Model calibration (global, NOT per-battery) ─────────────────────────
    if unc_metrics:
        ctx.append(
            f"MODEL CALIBRATION (global across all test batteries, "
            f"NOT specific to battery {battery_id}): "
            f"calibration score = {unc_metrics.get('calibration_score', 0):.3f} | "
            f"empirical 90% coverage = {unc_metrics.get('coverage_90_percent', 0):.1f}% | "
            f"average CI width across all test batteries = "
            f"{unc_metrics.get('mean_uncertainty_width', 0):.1f} cycles"
        )

    return "\n\n".join(ctx)


def main():
    print(f"Loading data from {BASE} ...")

    df         = pd.read_csv(BASE / "data/processed/cycle_features_with_rul.csv") \
                 if (BASE / "data/processed/cycle_features_with_rul.csv").exists() else pd.DataFrame()
    uncertainty = load_json(MDL / "uncertainty_estimates.json") or []
    anomalies   = load_json(MDL / "anomalies.json") or []
    survival    = pd.read_csv(MDL / "survival_risk_predictions.csv") \
                  if (MDL / "survival_risk_predictions.csv").exists() else pd.DataFrame()
    metrics     = load_json(TM  / "model_metrics.json") or {}
    feat_imp    = load_json(MDL / "feature_importance.json") or []
    hypotheses  = load_json(MDL / "degradation_hypotheses.json") or []
    counterfact = load_json(MDL / "counterfactual_examples.json") or []
    unc_metrics = load_json(MDL / "uncertainty_metrics.json") or {}
    drift       = load_json(MDL / "drift_report.json") or {}

    u_df = pd.DataFrame(uncertainty)
    a_df = pd.DataFrame(anomalies)

    RUL_COL = next(
        (c for c in ["rul_ensemble_mean", "rul_median", "rul_ensemble", "rul_ml", "rul_pred"]
         if not u_df.empty and c in u_df.columns), None
    )
    has_bands = RUL_COL is not None and {"rul_lower_5", "rul_upper_95"}.issubset(u_df.columns)

    print(f"Initialising LLM (first load may take ~30s for CUDA kernel compilation) ...")
    explainer = ExplanationBuilder(BASE)

    # Load existing results so we don't overwrite batteries we're not recomputing
    existing = load_json(OUT) or {}
    results  = dict(existing)

    for battery_id in BATTERIES_TO_COMPUTE:
        print(f"\n── {battery_id} ──────────────────────────────")
        try:
            extra_context = build_context(
                battery_id, df, u_df, a_df, survival, metrics,
                feat_imp, hypotheses, counterfact, drift, unc_metrics,
                RUL_COL, has_bands
            )
            query = (
                f"In exactly two short paragraphs (150 words total maximum), summarize the health of battery {battery_id}. "
                f"Paragraph 1: capacity state and fade percentage, RUL estimate with CI bounds, and failure risk level. "
                f"Paragraph 2: anomalies (if any), most likely degradation mechanism, and any contradictions between signals. "
                f"No headers, no bullet points, two paragraphs only."
            )
            answer, sources = explainer._rag.explain(query, extra_context=extra_context)
            results[battery_id] = {
                "answer":       answer,
                "sources":      sources,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            print(f"  OK — {len(answer)} chars")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[battery_id] = {"answer": None, "sources": [], "error": str(e),
                                   "generated_at": datetime.now().isoformat(timespec="seconds")}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {OUT}")
    print(f"Batteries in file: {sorted(results.keys())}")


if __name__ == "__main__":
    main()

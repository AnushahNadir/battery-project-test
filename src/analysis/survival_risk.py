# src/analysis/survival_risk.py
"""
Survival / Risk Modeling (Discrete-Time Hazard)

Goal (simple + robust):
- Learn a "risk of failure" signal over cycles using a discrete-time hazard model.
- Works per-cycle (row-level) instead of only per-battery.
- Produces:
    - hazard_prob: P(failure happens at this cycle | survived until previous cycle)
    - failure_prob_horizon: P(failure within next H cycles | at this cycle)

Definition of "failure / event":
- A battery "fails" when capacity <= eol_capacity_threshold for the FIRST time.
- If it never crosses, it's censored at its last observed cycle.

This is NOT causal. It's a statistical risk score for decision support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
DEFAULT_FEATURES = [
    "capacity",
    "temp_mean",
    "temp_max",
    "v_min",
    "v_mean",
    "i_mean",
    "i_min",
    "energy_j",
    "ah_est",
    "duration_s",
]


@dataclass
class SurvivalRiskMetrics:
    """Basic diagnostics for the hazard model."""
    n_rows: int
    n_batteries: int
    n_events: int
    event_rate: float
    horizon: int
    model_type: str
    notes: List[str]

    def to_dict(self) -> Dict:
        return {
            "n_rows": self.n_rows,
            "n_batteries": self.n_batteries,
            "n_events": self.n_events,
            "event_rate": round(self.event_rate, 6),
            "horizon": self.horizon,
            "model_type": self.model_type,
            "notes": self.notes,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_required_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for survival_risk: {missing}")


def _compute_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - event_cycle_per_battery: first cycle where capacity <= threshold (if exists)
      - event_observed_per_battery: True if crossing happened
      - event_at_cycle: True only at the failure cycle row
      - at_risk: True for cycles <= last observed cycle (always true for provided rows)
    """
    _ensure_required_cols(df, ["battery_id", "cycle_index", "capacity", "eol_capacity_threshold"])

    out = df.copy()
    out["capacity"] = pd.to_numeric(out["capacity"], errors="coerce")
    out["eol_capacity_threshold"] = pd.to_numeric(out["eol_capacity_threshold"], errors="coerce")
    out["cycle_index"] = pd.to_numeric(out["cycle_index"], errors="coerce")

    out = out.sort_values(["battery_id", "cycle_index"], kind="mergesort").reset_index(drop=True)

    # For each battery, find first crossing cycle
    event_cycle = {}
    event_observed = {}

    for bid, g in out.groupby("battery_id", sort=False):
        g2 = g.dropna(subset=["cycle_index"])
        if g2.empty:
            event_cycle[bid] = np.nan
            event_observed[bid] = False
            continue

        crossed = g2["capacity"] <= g2["eol_capacity_threshold"]
        if crossed.any():
            first_idx = crossed.idxmax()  # idxmax gives first True if bool
            # But idxmax returns original index; safer:
            first_row = g2.loc[crossed[crossed].index[0]]
            event_cycle[bid] = int(first_row["cycle_index"])
            event_observed[bid] = True
        else:
            event_cycle[bid] = int(g2["cycle_index"].max())
            event_observed[bid] = False  # censored

    out["event_cycle_per_battery"] = out["battery_id"].map(event_cycle)
    out["event_observed_per_battery"] = out["battery_id"].map(event_observed).astype(bool)

    out["event_at_cycle"] = (
        out["event_observed_per_battery"]
        & (out["cycle_index"].astype(float) == out["event_cycle_per_battery"].astype(float))
    )

    # All rows are "at risk" up to their cycle_index.
    out["at_risk"] = True

    return out


def _build_person_period_table(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Person-period table is basically the input df, but ensures:
      - features are numeric
      - event_at_cycle is binary target
    """
    _ensure_required_cols(df, ["battery_id", "cycle_index", "event_at_cycle"])
    out = df.copy()

    # Coerce features
    for c in feature_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan  # allow missing features; will be imputed

    out["y_event"] = out["event_at_cycle"].astype(int)

    # Remove rows with missing cycle_index or battery_id
    out = out.dropna(subset=["battery_id", "cycle_index"]).copy()
    out["cycle_index"] = out["cycle_index"].astype(int)

    return out


def _fit_logistic_hazard(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, str, List[str]]:
    """
    Fits a simple logistic regression.
    Prefers sklearn if available, otherwise uses a tiny deterministic gradient descent.

    Returns:
      - coef vector
      - intercept
      - model_type string
      - notes
    """
    notes: List[str] = []

    # Try sklearn first (clean + stable)
    try:
        from sklearn.linear_model import LogisticRegression

        # L2 regularization helps stability; deterministic with fixed solver + random_state
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )
        clf.fit(X, y)
        coef = clf.coef_.reshape(-1).astype(float)
        intercept = float(clf.intercept_[0])
        return coef, intercept, "sklearn_logistic_regression", notes

    except Exception as e:
        notes.append(f"sklearn not available or failed ({e}); using fallback gradient descent.")

    # Fallback: deterministic gradient descent (very simple)
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0

    lr = 0.05
    reg = 1e-3
    steps = 800

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        # gradients
        grad_w = (X.T @ (p - y)) / n + reg * w
        grad_b = float(np.mean(p - y))

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, "gd_logistic_fallback", notes


def _impute_and_standardize(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Impute missing with median, then standardize (z-score) for stable logistic training.
    Returns transformed df + medians + stds.
    """
    Xdf = df[feature_cols].copy()

    medians = {}
    stds = {}
    for c in feature_cols:
        med = float(np.nanmedian(Xdf[c].values)) if np.isfinite(np.nanmedian(Xdf[c].values)) else 0.0
        Xdf[c] = Xdf[c].fillna(med)
        medians[c] = med

        sd = float(np.nanstd(Xdf[c].values))
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        stds[c] = sd
        Xdf[c] = (Xdf[c] - medians[c]) / stds[c]

    out = df.copy()
    for c in feature_cols:
        out[c] = Xdf[c]

    return out, medians, stds


def _predict_hazard(df: pd.DataFrame, feature_cols: List[str], coef: np.ndarray, intercept: float) -> np.ndarray:
    X = df[feature_cols].values.astype(float)
    z = X @ coef + intercept
    return _sigmoid(z)


def _failure_prob_within_horizon(hazards: np.ndarray, horizon: int) -> np.ndarray:
    """
    Given per-cycle hazards for a single battery sequence,
    compute P(event within next H cycles) at each cycle.

    For discrete hazard h_t:
      survival over next k steps = Π(1 - h_{t+i})
      failure within H = 1 - Π_{i=1..H}(1 - h_{t+i})
    """
    n = len(hazards)
    out = np.zeros(n, dtype=float)

    # Precompute (1 - h)
    q = 1.0 - np.clip(hazards, 1e-8, 1.0 - 1e-8)

    for t in range(n):
        end = min(n, t + horizon + 1)
        # next cycles: t+1 .. end-1
        if t + 1 >= end:
            out[t] = 0.0
            continue
        surv = float(np.prod(q[t + 1:end]))
        out[t] = 1.0 - surv

    return out


def _battery_number(battery_id: str) -> Optional[int]:
    bid = str(battery_id).strip().upper()
    if bid.startswith("B") and bid[1:].isdigit():
        return int(bid[1:])
    return None


def _temp_group(battery_id: str) -> str:
    num = _battery_number(battery_id)
    if num is None:
        return "room"
    if 41 <= num <= 56:
        return "cold"
    if 29 <= num <= 32 or 38 <= num <= 40:
        return "hot"
    return "room"


def _build_km_curve(durations: np.ndarray, observed: np.ndarray) -> Dict[str, object]:
    """
    Build Kaplan-Meier curve from battery-level duration/event arrays.
    Returns survival S(t) for integer cycles t >= 0.
    """
    if len(durations) == 0:
        return {
            "max_t": 0,
            "survival_by_t": {0: 1.0},
            "n_at_risk_by_t": {0: 0},
            "d_events_by_t": {0: 0},
        }

    durs = np.asarray(durations, dtype=int)
    obs = np.asarray(observed, dtype=bool)
    max_t = int(max(1, np.max(durs)))

    survival_by_t: Dict[int, float] = {0: 1.0}
    n_at_risk_by_t: Dict[int, int] = {}
    d_events_by_t: Dict[int, int] = {}

    s_prev = 1.0
    for t in range(1, max_t + 1):
        n_at_risk = int(np.sum(durs >= t))
        d_events = int(np.sum((durs == t) & obs))
        n_at_risk_by_t[t] = n_at_risk
        d_events_by_t[t] = d_events
        if n_at_risk > 0:
            s_prev = s_prev * (1.0 - d_events / n_at_risk)
        survival_by_t[t] = float(np.clip(s_prev, 0.0, 1.0))

    return {
        "max_t": max_t,
        "survival_by_t": survival_by_t,
        "n_at_risk_by_t": n_at_risk_by_t,
        "d_events_by_t": d_events_by_t,
    }


def _km_survival_at(curve: Dict[str, object], t: int) -> float:
    max_t = int(curve["max_t"])
    surv = curve["survival_by_t"]
    if t <= 0:
        return 1.0
    if t > max_t:
        return float(surv[max_t])
    return float(surv[t])


def _km_failure_within_horizon(curve: Dict[str, object], cycle_index: int, horizon: int) -> tuple[float, float]:
    s_now = _km_survival_at(curve, int(cycle_index))
    s_next = _km_survival_at(curve, int(cycle_index) + 1)
    s_h = _km_survival_at(curve, int(cycle_index) + int(horizon))

    if s_now <= 0:
        return 1.0, 1.0

    hazard_prob = float(np.clip(1.0 - (s_next / s_now), 0.0, 1.0))
    failure_prob_h = float(np.clip(1.0 - (s_h / s_now), 0.0, 1.0))
    return hazard_prob, failure_prob_h


def run_survival_risk(
    df: pd.DataFrame,
    output_dir: Path,
    feature_cols: Optional[List[str]] = None,
    horizon: Optional[int] = None,
) -> Tuple[pd.DataFrame, SurvivalRiskMetrics]:
    """
    Main entrypoint.

    Input df should be your merged cycle table (cycle_features_with_rul.csv),
    containing at least:
      - battery_id, cycle_index, capacity, eol_capacity_threshold

    Saves:
      - survival_risk_predictions.csv
      - survival_risk_metrics.json
      - survival_risk_report.md
      - survival_risk_model.json (coefficients + preprocessing params)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_config()
    horizon = int(cfg.survival.horizon_cycles if horizon is None else horizon)
    method = str(cfg.survival.method).strip().lower()

    feats = feature_cols or DEFAULT_FEATURES

    # 1) Compute event flags
    flagged = _compute_event_flags(df)

    # 2) Build person-period table
    pp = _build_person_period_table(flagged, feats)
    pp["temp_group"] = pp["battery_id"].astype(str).apply(_temp_group)

    notes: List[str] = []
    model_blob: Dict[str, object]

    if method == "kaplan_meier":
        model_type = "kaplan_meier_grouped"
        pp_km = pp.copy()
        # Build battery-level event table for KM fit.
        batt_table = (
            pp_km[["battery_id", "event_cycle_per_battery", "event_observed_per_battery", "temp_group"]]
            .drop_duplicates(subset=["battery_id"])
            .copy()
        )
        batt_table["event_cycle_per_battery"] = pd.to_numeric(
            batt_table["event_cycle_per_battery"], errors="coerce"
        ).fillna(1).astype(int)
        batt_table["event_observed_per_battery"] = batt_table["event_observed_per_battery"].astype(bool)

        curves: Dict[str, Dict[str, object]] = {}
        for grp, g in batt_table.groupby("temp_group", sort=True):
            curves[str(grp)] = _build_km_curve(
                durations=g["event_cycle_per_battery"].values.astype(int),
                observed=g["event_observed_per_battery"].values.astype(bool),
            )
        global_curve = _build_km_curve(
            durations=batt_table["event_cycle_per_battery"].values.astype(int),
            observed=batt_table["event_observed_per_battery"].values.astype(bool),
        )

        hazards = np.zeros(len(pp_km), dtype=float)
        fp_h = np.zeros(len(pp_km), dtype=float)
        for idx, row in pp_km.iterrows():
            grp = str(row["temp_group"])
            c = int(row["cycle_index"])
            curve = curves.get(grp, global_curve)
            hz, fp = _km_failure_within_horizon(curve, c, horizon=horizon)
            hazards[idx] = hz
            fp_h[idx] = fp

        pp_km["hazard_prob"] = np.clip(hazards, 0.0, 1.0)
        pp_km["failure_prob_horizon"] = np.clip(fp_h, 0.0, 1.0)
        pp_km["risk_category"] = pp_km["failure_prob_horizon"].apply(lambda p: cfg.risk.risk_category(float(p)))
        preds = pp_km

        model_blob = {
            "model_type": model_type,
            "method": method,
            "grouping": "temp_group",
            "horizon": horizon,
            "notes": notes,
            "curves": {
                grp: {
                    "max_t": int(curve["max_t"]),
                    "survival_by_t": {str(k): round(float(v), 6) for k, v in curve["survival_by_t"].items()},
                }
                for grp, curve in curves.items()
            },
            "_disclaimer": "Kaplan-Meier survival estimates are statistical, not causal.",
        }
    else:
        # Fallback/alternate method: logistic discrete-time hazard.
        pp_std, medians, stds = _impute_and_standardize(pp, feats)
        X = pp_std[feats].values.astype(float)
        y = pp_std["y_event"].values.astype(int)

        coef, intercept, model_type, notes = _fit_logistic_hazard(X, y)
        if method == "cox_ph":
            notes.append("Configured method='cox_ph' is approximated with logistic discrete-time hazard in this version.")
            model_type = "cox_ph_proxy_logistic_hazard"
        pp_std["hazard_prob"] = _predict_hazard(pp_std, feats, coef, intercept)

        pp_std["failure_prob_horizon"] = 0.0
        for bid, g in pp_std.groupby("battery_id", sort=False):
            g_sorted = g.sort_values("cycle_index", kind="mergesort")
            fp = _failure_prob_within_horizon(g_sorted["hazard_prob"].values.astype(float), horizon=horizon)
            pp_std.loc[g_sorted.index, "failure_prob_horizon"] = fp

        pp_std["risk_category"] = pp_std["failure_prob_horizon"].apply(lambda p: cfg.risk.risk_category(float(p)))
        preds = pp_std

        model_blob = {
            "model_type": model_type,
            "method": method,
            "feature_cols": feats,
            "coef": {f: float(w) for f, w in zip(feats, coef)},
            "intercept": float(intercept),
            "medians": medians,
            "stds": stds,
            "horizon": horizon,
            "notes": notes,
            "_disclaimer": "This is a statistical risk score (discrete-time hazard). Not causal.",
        }

    n_rows = len(preds)
    n_batt = int(preds["battery_id"].nunique())
    n_events = int(preds["event_at_cycle"].sum()) if "event_at_cycle" in preds.columns else 0
    event_rate = float(n_events / n_rows) if n_rows else 0.0

    metrics = SurvivalRiskMetrics(
        n_rows=n_rows,
        n_batteries=n_batt,
        n_events=n_events,
        event_rate=event_rate,
        horizon=horizon,
        model_type=model_type,
        notes=notes,
    )

    pred_cols = [
        "battery_id",
        "cycle_index",
        "hazard_prob",
        "failure_prob_horizon",
        "risk_category",
        "event_at_cycle",
        "event_cycle_per_battery",
        "event_observed_per_battery",
        "temp_group",
    ]
    pred_cols = [c for c in pred_cols if c in preds.columns]
    preds_out = preds[pred_cols].sort_values(["battery_id", "cycle_index"], kind="mergesort")
    preds_path = output_dir / "survival_risk_predictions.csv"
    preds_out.to_csv(preds_path, index=False)

    (output_dir / "survival_risk_model.json").write_text(json.dumps(model_blob, indent=2), encoding="utf-8")
    (output_dir / "survival_risk_metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")

    report_lines = [
        "# Survival Risk Report",
        "",
        f"**Model:** {model_type}",
        f"**Configured method:** {method}",
        f"**Horizon:** {horizon} cycles",
        "",
        "## What this means (simple)",
        "",
        "- `hazard_prob` = chance the battery fails *at this cycle* (given it survived before).",
        f"- `failure_prob_horizon` = chance the battery fails within the next **{horizon} cycles**.",
        "- This is a risk signal, not a causal explanation.",
        "",
        "## Data summary",
        "",
        f"- Rows: {metrics.n_rows}",
        f"- Batteries: {metrics.n_batteries}",
        f"- Events observed: {metrics.n_events}",
        f"- Event rate: {metrics.event_rate:.4%}",
        "",
        "## Risk distribution (by row)",
        "",
    ]
    dist = preds_out["risk_category"].value_counts(dropna=False).to_dict()
    total = float(len(preds_out)) if len(preds_out) else 1.0
    report_lines.append("| Category | Count | % |")
    report_lines.append("|---|---:|---:|")
    for k in ["LOW", "MEDIUM", "HIGH"]:
        c = int(dist.get(k, 0))
        report_lines.append(f"| {k} | {c} | {100.0*c/total:.1f}% |")

    if method != "kaplan_meier":
        report_lines.extend([
            "",
            "## Top signals (model weights)",
            "",
            "> Larger absolute weight = model uses that feature more for risk scoring.",
            "",
            "| Feature | Weight |",
            "|---|---:|",
        ])
        weights_sorted = sorted(model_blob.get("coef", {}).items(), key=lambda x: abs(x[1]), reverse=True)
        for f, w in weights_sorted[:10]:
            report_lines.append(f"| `{f}` | {float(w):.4f} |")
    else:
        report_lines.extend([
            "",
            "## Kaplan-Meier Groups",
            "",
            "- Survival curves are estimated per temperature group (`room`, `hot`, `cold`).",
            "- Horizon failure risk is computed as `1 - S(t+h)/S(t)`.",
        ])

    if notes:
        report_lines.extend(["", "## Notes", ""] + [f"- {n}" for n in notes])

    (output_dir / "survival_risk_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return preds_out, metrics


if __name__ == "__main__":
    # Example local run (adjust paths if needed)
    base = Path("d:/Energy Project/code/battery-project3")
    df = pd.read_csv(base / "data/processed/cycle_features_with_rul.csv")

    preds, metrics = run_survival_risk(
        df=df,
        output_dir=base / "data/processed/modeling",
        feature_cols=DEFAULT_FEATURES,
        horizon=get_config().survival.horizon_cycles,
    )

    logger.info("=== SURVIVAL RISK DONE ===")
    logger.info(json.dumps(metrics.to_dict(), indent=2))
    logger.info(f"Saved: {base / 'data/processed/modeling/survival_risk_predictions.csv'}")

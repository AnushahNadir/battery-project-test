# src/modeling/reliability_audit.py
# Updated: 2026-04-16
"""
Reliability Audit Layer — Phase 4 implementation.

Produces a deterministic per-battery verdict: PASS / WARNING / FAIL.

Inputs (per battery, aggregated over last K observed cycles):
  - predicted RUL
  - uncertainty interval width
  - near-term failure probability
  - anomaly flag count

Rules (all thresholds sourced from configs/pipeline.yaml → audit section):
  PASS    — low uncertainty, low risk, no anomalies, consistent prediction
  WARNING — moderate uncertainty OR moderate risk OR anomaly present
  FAIL    — high uncertainty OR high failure risk OR major anomaly count

Output artifacts:
  - data/processed/modeling/per_battery_results.csv
  - data/processed/modeling/metrics_audit.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Threshold defaults (overridden by configs/pipeline.yaml → audit) ──────────

@dataclass
class AuditConfig:
    # Aggregation window: last K cycles per battery
    observation_window: int = 10
    # Uncertainty interval width thresholds (in cycles)
    warn_interval_width: float = 80.0   # >= this → WARNING
    fail_interval_width: float = 120.0  # >= this → FAIL
    # Near-term failure probability thresholds (0–1)
    warn_risk: float = 0.30             # >= this → WARNING
    fail_risk: float = 0.70             # >= this → FAIL
    # Anomaly count thresholds (over window)
    warn_anomaly_count: int = 1         # >= this → WARNING
    fail_anomaly_count: int = 3         # >= this → FAIL


@dataclass
class BatteryAuditResult:
    battery_id: str
    dataset: str
    temperature_group: str
    actual_eol_cycle: int | None
    n_cycles_observed: int
    # Prediction
    mean_predicted_rul: float
    last_predicted_rul: float
    mean_abs_error: float | None
    # Uncertainty
    mean_interval_width: float
    last_interval_width: float
    # Risk
    mean_failure_prob: float
    max_failure_prob: float
    # Anomaly
    anomaly_count: int
    # Confidence interval bounds (last cycle)
    interval_lower: float
    interval_upper: float
    # Audit
    audit_label: str       # PASS / WARNING / FAIL
    audit_reasons: list[str] = field(default_factory=list)


def _get_temp_group(battery_id: str) -> str:
    bid = str(battery_id).strip().upper()
    if bid.startswith("CS"):
        return "room"
    digits = "".join(c for c in bid if c.isdigit())
    if not digits:
        return "room"
    n = int(digits)
    if 41 <= n <= 56:
        return "cold"
    if (29 <= n <= 32) or (38 <= n <= 40):
        return "hot"
    return "room"


def _assign_verdict(
    interval_width: float,
    failure_prob: float,
    anomaly_count: int,
    cfg: AuditConfig,
) -> tuple[str, list[str]]:
    """
    Deterministic PASS / WARNING / FAIL assignment.
    Returns (label, reasons).
    """
    reasons: list[str] = []

    # FAIL conditions (any one is sufficient)
    if interval_width >= cfg.fail_interval_width:
        reasons.append(f"interval_width={interval_width:.1f} >= fail threshold {cfg.fail_interval_width:.1f}")
    if failure_prob >= cfg.fail_risk:
        reasons.append(f"failure_prob={failure_prob:.3f} >= fail threshold {cfg.fail_risk:.2f}")
    if anomaly_count >= cfg.fail_anomaly_count:
        reasons.append(f"anomaly_count={anomaly_count} >= fail threshold {cfg.fail_anomaly_count}")

    if reasons:
        return "FAIL", reasons

    # WARNING conditions (any one is sufficient)
    if interval_width >= cfg.warn_interval_width:
        reasons.append(f"interval_width={interval_width:.1f} >= warn threshold {cfg.warn_interval_width:.1f}")
    if failure_prob >= cfg.warn_risk:
        reasons.append(f"failure_prob={failure_prob:.3f} >= warn threshold {cfg.warn_risk:.2f}")
    if anomaly_count >= cfg.warn_anomaly_count:
        reasons.append(f"anomaly_count={anomaly_count} >= warn threshold {cfg.warn_anomaly_count}")

    if reasons:
        return "WARNING", reasons

    return "PASS", ["all metrics within safe thresholds"]


def run_reliability_audit(
    features_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    output_dir: Path,
    audit_cfg: AuditConfig | None = None,
    rul_col: str = "rul_ensemble_mean",
) -> tuple[list[BatteryAuditResult], dict[str, Any]]:
    """
    Run the per-battery Reliability Audit Layer.

    Parameters
    ----------
    features_df   : cycle_features_with_rul.csv (has battery_id, RUL, cycle_index)
    uncertainty_df: uncertainty_estimates.json as DataFrame (has rul_lower_5, rul_upper_95)
    survival_df   : survival_risk_predictions.csv (has failure_prob_horizon)
    anomaly_df    : anomalies.json as DataFrame (has battery_id, cycle_index)
    output_dir    : where to write per_battery_results.csv and metrics_audit.json
    audit_cfg     : threshold config (uses defaults if None)
    rul_col       : column name for predicted RUL in uncertainty_df
    """
    cfg = audit_cfg or AuditConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise types
    for df_ in [features_df, uncertainty_df, survival_df, anomaly_df]:
        if not df_.empty and "battery_id" in df_.columns:
            df_["battery_id"] = df_["battery_id"].astype(str)

    # Find the RUL column
    rul_candidates = [rul_col, "rul_ensemble_mean", "rul_median", "rul_ensemble", "rul_ml", "rul_pred"]
    actual_rul_col = next((c for c in rul_candidates if not uncertainty_df.empty and c in uncertainty_df.columns), None)

    has_lower = not uncertainty_df.empty and "rul_lower_5" in uncertainty_df.columns
    has_upper = not uncertainty_df.empty and "rul_upper_95" in uncertainty_df.columns
    has_survival = not survival_df.empty and "failure_prob_horizon" in survival_df.columns

    batteries = sorted(features_df["battery_id"].dropna().unique()) if not features_df.empty else []

    results: list[BatteryAuditResult] = []

    for bid in batteries:
        feat_b = features_df[features_df["battery_id"] == bid].sort_values("cycle_index")
        unc_b  = uncertainty_df[uncertainty_df["battery_id"] == bid].sort_values("cycle_index") \
                 if not uncertainty_df.empty and "battery_id" in uncertainty_df.columns else pd.DataFrame()
        surv_b = survival_df[survival_df["battery_id"] == bid].sort_values("cycle_index") \
                 if not survival_df.empty and "battery_id" in survival_df.columns else pd.DataFrame()
        anom_b = anomaly_df[anomaly_df["battery_id"] == bid] \
                 if not anomaly_df.empty and "battery_id" in anomaly_df.columns else pd.DataFrame()

        # Window: last K cycles
        K = cfg.observation_window
        unc_w  = unc_b.iloc[-K:]  if len(unc_b)  > 0 else unc_b
        surv_w = surv_b.iloc[-K:] if len(surv_b) > 0 else surv_b

        # EOL
        eol_cycle = None
        if "eol_cycle" in feat_b.columns and len(feat_b):
            try:
                eol_cycle = int(feat_b["eol_cycle"].iloc[0])
            except Exception:
                pass

        # Dataset
        dataset = "unknown"
        if "source_dataset" in feat_b.columns and len(feat_b):
            dataset = str(feat_b["source_dataset"].iloc[0])
        elif str(bid).upper().startswith("CS"):
            dataset = "CALCE"
        else:
            dataset = "NASA"

        # Predicted RUL
        mean_pred_rul = float("nan")
        last_pred_rul = float("nan")
        if actual_rul_col and not unc_w.empty and actual_rul_col in unc_w.columns:
            vals = pd.to_numeric(unc_w[actual_rul_col], errors="coerce").dropna()
            if len(vals):
                mean_pred_rul = float(vals.mean())
                last_pred_rul = float(vals.iloc[-1])

        # Absolute error (vs true RUL)
        mean_abs_error = None
        if actual_rul_col and "RUL" in feat_b.columns and not unc_w.empty and actual_rul_col in unc_w.columns:
            merged = unc_w[["cycle_index", actual_rul_col]].merge(
                feat_b[["cycle_index", "RUL"]], on="cycle_index", how="inner"
            )
            if len(merged):
                pred_vals = pd.to_numeric(merged[actual_rul_col], errors="coerce")
                true_vals = pd.to_numeric(merged["RUL"], errors="coerce")
                err = (pred_vals - true_vals).abs().dropna()
                if len(err):
                    mean_abs_error = float(err.mean())

        # Uncertainty interval width
        mean_iw = float("nan")
        last_iw = float("nan")
        lower   = float("nan")
        upper   = float("nan")
        if has_lower and has_upper and not unc_w.empty:
            lo = pd.to_numeric(unc_w["rul_lower_5"],  errors="coerce")
            hi = pd.to_numeric(unc_w["rul_upper_95"], errors="coerce")
            widths = (hi - lo).dropna()
            if len(widths):
                mean_iw = float(widths.mean())
                last_iw = float(widths.iloc[-1])
            if len(unc_w):
                lower = float(pd.to_numeric(unc_w["rul_lower_5"],  errors="coerce").iloc[-1])
                upper = float(pd.to_numeric(unc_w["rul_upper_95"], errors="coerce").iloc[-1])

        # Failure probability
        mean_fp = 0.0
        max_fp  = 0.0
        if has_survival and not surv_w.empty:
            fp = pd.to_numeric(surv_w["failure_prob_horizon"], errors="coerce").dropna()
            if len(fp):
                mean_fp = float(fp.mean())
                max_fp  = float(fp.max())

        # Anomaly count in window
        anom_count = 0
        if not anom_b.empty and "cycle_index" in anom_b.columns and not unc_w.empty:
            window_cycles = set(unc_w["cycle_index"].astype(str).tolist())
            anom_cycles   = set(anom_b["cycle_index"].astype(str).tolist())
            anom_count = len(window_cycles & anom_cycles)

        # Verdict — use max_fp and last_iw as the primary signals
        audit_iw = last_iw if np.isfinite(last_iw) else mean_iw if np.isfinite(mean_iw) else 0.0
        label, reasons = _assign_verdict(audit_iw, max_fp, anom_count, cfg)

        results.append(BatteryAuditResult(
            battery_id=bid,
            dataset=dataset,
            temperature_group=_get_temp_group(bid),
            actual_eol_cycle=eol_cycle,
            n_cycles_observed=len(feat_b),
            mean_predicted_rul=mean_pred_rul,
            last_predicted_rul=last_pred_rul,
            mean_abs_error=mean_abs_error,
            mean_interval_width=mean_iw,
            last_interval_width=last_iw,
            mean_failure_prob=mean_fp,
            max_failure_prob=max_fp,
            anomaly_count=anom_count,
            interval_lower=lower,
            interval_upper=upper,
            audit_label=label,
            audit_reasons=reasons,
        ))

    # ── Write per_battery_results.csv ─────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({
            "battery_id":          r.battery_id,
            "dataset":             r.dataset,
            "temperature_group":   r.temperature_group,
            "actual_eol_cycle":    r.actual_eol_cycle,
            "n_cycles_observed":   r.n_cycles_observed,
            "predicted_rul":       round(r.last_predicted_rul, 2) if np.isfinite(r.last_predicted_rul) else None,
            "mean_abs_error":      round(r.mean_abs_error, 2) if r.mean_abs_error is not None else None,
            "interval_lower":      round(r.interval_lower, 2) if np.isfinite(r.interval_lower) else None,
            "interval_upper":      round(r.interval_upper, 2) if np.isfinite(r.interval_upper) else None,
            "interval_width":      round(r.last_interval_width, 2) if np.isfinite(r.last_interval_width) else None,
            "failure_risk":        round(r.max_failure_prob, 4),
            "anomaly_flag":        r.anomaly_count > 0,
            "anomaly_count":       r.anomaly_count,
            "audit_label":         r.audit_label,
            "audit_reasons":       "; ".join(r.audit_reasons),
        })
    per_bat_df = pd.DataFrame(rows)
    per_bat_path = output_dir / "per_battery_results.csv"
    per_bat_df.to_csv(per_bat_path, index=False)
    logger.info(f"Per-battery results -> {per_bat_path}  ({len(per_bat_df)} batteries)")

    # ── Write metrics_audit.json ──────────────────────────────────────────────
    label_counts = per_bat_df["audit_label"].value_counts().to_dict() if len(per_bat_df) else {}
    n_total = len(per_bat_df)

    # RMSE per audit label (only for batteries with mean_abs_error)
    rmse_by_label: dict[str, float] = {}
    for label in ["PASS", "WARNING", "FAIL"]:
        sub = per_bat_df[(per_bat_df["audit_label"] == label) & per_bat_df["mean_abs_error"].notna()]
        if len(sub):
            rmse_by_label[label] = round(float(np.sqrt((sub["mean_abs_error"] ** 2).mean())), 4)

    # Severe error catch rate: fraction of batteries with MAE > 100 cycles that are WARNING or FAIL
    if "mean_abs_error" in per_bat_df.columns:
        severe = per_bat_df[per_bat_df["mean_abs_error"].notna() & (per_bat_df["mean_abs_error"] > 100)]
        if len(severe):
            caught = severe[severe["audit_label"].isin(["WARNING", "FAIL"])]
            severe_catch_rate = round(len(caught) / len(severe), 4)
        else:
            severe_catch_rate = None
    else:
        severe_catch_rate = None

    metrics_audit = {
        "n_batteries": n_total,
        "label_counts": label_counts,
        "label_rates": {k: round(v / n_total, 4) for k, v in label_counts.items()} if n_total else {},
        "rmse_by_audit_label": rmse_by_label,
        "severe_error_catch_rate": severe_catch_rate,
        "audit_config": {
            "observation_window": cfg.observation_window,
            "warn_interval_width": cfg.warn_interval_width,
            "fail_interval_width": cfg.fail_interval_width,
            "warn_risk": cfg.warn_risk,
            "fail_risk": cfg.fail_risk,
            "warn_anomaly_count": cfg.warn_anomaly_count,
            "fail_anomaly_count": cfg.fail_anomaly_count,
        },
    }
    audit_path = output_dir / "metrics_audit.json"
    audit_path.write_text(json.dumps(metrics_audit, indent=2), encoding="utf-8")
    logger.info(f"Audit metrics -> {audit_path}")

    return results, metrics_audit

# scripts/run_ablation.py
# Updated: 2026-04-16
"""
Ablation Study — Phase 5, Experiment 5.

Runs four ablation configurations to show that each reliability module
adds measurable value beyond raw prediction:

  Config A: XGBoost only (no uncertainty, no risk, no audit)
  Config B: XGBoost + conformal uncertainty
  Config C: XGBoost + uncertainty + survival risk
  Config D: Full system (XGBoost + uncertainty + risk + audit)

For each config reports:
  - RMSE / MAE on test batteries
  - Conformal coverage (if uncertainty enabled)
  - % of bad predictions caught by WARNING or FAIL (if audit enabled)
  - Severe-error catch rate

Usage:
    python -m scripts.run_ablation
    python scripts/run_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.modeling.ml_model import FEATURE_COLUMNS, XGBoostRULModel
from src.modeling.conformal import ConformalCalibrator, select_calibration_batteries
from src.analysis.survival_risk import DEFAULT_FEATURES, run_survival_risk
from src.modeling.reliability_audit import AuditConfig, run_reliability_audit
from src.config import get_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATA_PATH  = BASE / "data/processed/cycle_features_with_rul.csv"
OUTPUT_DIR = BASE / "data/processed/modeling"


@dataclass
class AblationResult:
    config:              str
    description:         str
    rmse:                float
    mae:                 float
    coverage:            float | None
    mean_interval_width: float | None
    pass_rate:           float | None
    warning_rate:        float | None
    fail_rate:           float | None
    severe_catch_rate:   float | None


def _get_temp_group(bid: str) -> str:
    bid = str(bid).strip().upper()
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


def _stratified_split(batteries, train_ratio=0.8, seed=42):
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    groups = defaultdict(list)
    for b in sorted(batteries):
        groups[_get_temp_group(b)].append(b)
    train_ids, test_ids = [], []
    for g in sorted(groups):
        bats = groups[g][:]
        rng.shuffle(bats)
        n = len(bats)
        if n <= 1:
            train_ids.extend(bats)
            continue
        n_train = max(1, min(int(round(train_ratio * n)), n - 1))
        train_ids.extend(bats[:n_train])
        test_ids.extend(bats[n_train:])
    if not test_ids and len(train_ids) > 1:
        test_ids.append(train_ids.pop())
    return sorted(train_ids), sorted(test_ids)


def run_ablation() -> list[AblationResult]:
    cfg = get_config()

    if not DATA_PATH.exists():
        print(f"[ERROR] Data not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df["battery_id"] = df["battery_id"].astype(str)
    batteries = sorted(df["battery_id"].unique())

    train_batteries, test_batteries = _stratified_split(
        batteries, train_ratio=cfg.split.train_ratio, seed=cfg.split.random_seed
    )
    cal_batteries, actual_train = select_calibration_batteries(
        train_batteries=train_batteries,
        n_cal=cfg.split.n_cal_batteries,
        random_seed=cfg.split.random_seed,
    )

    train_df = df[df["battery_id"].isin(actual_train)].copy()
    cal_df   = df[df["battery_id"].isin(cal_batteries)].copy()
    test_df  = df[df["battery_id"].isin(test_batteries)].copy()

    if test_df.empty or "RUL" not in test_df.columns:
        print("[ERROR] Test set empty or missing RUL column.")
        sys.exit(1)

    actual_rul = test_df["RUL"].values.astype(float)

    print("=" * 65)
    print("ABLATION STUDY")
    print("=" * 65)
    print(f"Train: {len(actual_train)} batteries | Cal: {len(cal_batteries)} | Test: {len(test_batteries)}")
    print()

    results: list[AblationResult] = []
    ablation_output = OUTPUT_DIR / "ablation"
    ablation_output.mkdir(parents=True, exist_ok=True)

    # ── Config A: XGBoost only ────────────────────────────────────────────────
    print("[A] XGBoost only...")
    model_a = XGBoostRULModel()
    model_a.fit(train_df, quantiles=False)
    preds_a = model_a.predict(test_df).astype(float)
    valid_a = np.isfinite(preds_a) & np.isfinite(actual_rul)
    rmse_a  = float(np.sqrt(np.mean((preds_a[valid_a] - actual_rul[valid_a]) ** 2)))
    mae_a   = float(np.mean(np.abs(preds_a[valid_a] - actual_rul[valid_a])))
    print(f"   RMSE={rmse_a:.2f}  MAE={mae_a:.2f}")

    results.append(AblationResult(
        config="A", description="XGBoost only",
        rmse=round(rmse_a, 4), mae=round(mae_a, 4),
        coverage=None, mean_interval_width=None,
        pass_rate=None, warning_rate=None, fail_rate=None,
        severe_catch_rate=None,
    ))

    # ── Config B: XGBoost + conformal uncertainty ─────────────────────────────
    print("[B] XGBoost + conformal uncertainty...")
    model_b = XGBoostRULModel()
    model_b.fit(train_df, quantiles=True)
    preds_b = model_b.predict(test_df).astype(float)
    valid_b = np.isfinite(preds_b) & np.isfinite(actual_rul)
    rmse_b  = float(np.sqrt(np.mean((preds_b[valid_b] - actual_rul[valid_b]) ** 2)))
    mae_b   = float(np.mean(np.abs(preds_b[valid_b] - actual_rul[valid_b])))

    # Conformal calibration
    calibrator_b = ConformalCalibrator(coverage=cfg.conformal.coverage, rul_col="RUL")
    train_df_lobo = df[df["battery_id"].isin(train_batteries)].copy()
    calibrator_b.fit(cal_df, model_b, train_df=train_df_lobo, actual_train_batteries=actual_train)
    conf_report_b = calibrator_b.coverage_report(test_df, model_b)
    coverage_b    = float(conf_report_b.get("overall_empirical_coverage", 0.0))

    # Mean interval width
    qpreds_b = model_b.predict_quantiles(test_df)
    iw_b = None
    lo_col = next((c for c in ["q05", "q_0.05"] if c in qpreds_b.columns), None)
    hi_col = next((c for c in ["q95", "q_0.95"] if c in qpreds_b.columns), None)
    if lo_col and hi_col:
        iw_b = float((qpreds_b[hi_col] - qpreds_b[lo_col]).mean())

    iw_str = f"{iw_b:.1f}" if iw_b is not None else "N/A"
    print(f"   RMSE={rmse_b:.2f}  MAE={mae_b:.2f}  Coverage={coverage_b*100:.1f}%  AvgWidth={iw_str}")

    results.append(AblationResult(
        config="B", description="XGBoost + conformal uncertainty",
        rmse=round(rmse_b, 4), mae=round(mae_b, 4),
        coverage=round(coverage_b, 4), mean_interval_width=round(iw_b, 2) if iw_b else None,
        pass_rate=None, warning_rate=None, fail_rate=None,
        severe_catch_rate=None,
    ))

    # ── Config C: XGBoost + uncertainty + survival risk ───────────────────────
    print("[C] XGBoost + uncertainty + survival risk...")
    # Same model as B, add risk
    surv_dir_c = ablation_output / "config_c"
    surv_dir_c.mkdir(exist_ok=True)
    try:
        _, surv_metrics_c = run_survival_risk(
            df=test_df, output_dir=surv_dir_c,
            feature_cols=DEFAULT_FEATURES, horizon=cfg.risk.horizon_cycles,
        )
        risk_ok_c = True
        print(f"   Survival risk computed for {surv_metrics_c.n_batteries} batteries")
    except Exception as e:
        risk_ok_c = False
        print(f"   Survival risk failed: {e}")

    print(f"   RMSE={rmse_b:.2f}  MAE={mae_b:.2f}  Coverage={coverage_b*100:.1f}%  Risk={'OK' if risk_ok_c else 'FAIL'}")

    results.append(AblationResult(
        config="C", description="XGBoost + uncertainty + survival risk",
        rmse=round(rmse_b, 4), mae=round(mae_b, 4),
        coverage=round(coverage_b, 4), mean_interval_width=round(iw_b, 2) if iw_b else None,
        pass_rate=None, warning_rate=None, fail_rate=None,
        severe_catch_rate=None,
    ))

    # ── Config D: Full system (+ audit layer) ─────────────────────────────────
    print("[D] Full system (XGBoost + uncertainty + risk + audit)...")
    surv_dir_d = ablation_output / "config_d"
    surv_dir_d.mkdir(exist_ok=True)

    # Build uncertainty_df from quantile predictions + calibrator adjustments
    q_preds = model_b.predict_quantiles(test_df)
    _lo_col = next((c for c in ["q05", "q_0.05"] if c in q_preds.columns), None)
    _hi_col = next((c for c in ["q95", "q_0.95"] if c in q_preds.columns), None)
    unc_rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        lo = float(q_preds.iloc[i][_lo_col]) if _lo_col else float("nan")
        hi = float(q_preds.iloc[i][_hi_col]) if _hi_col else float("nan")
        unc_rows.append({
            "battery_id":       row["battery_id"],
            "cycle_index":      row.get("cycle_index", i),
            "rul_ensemble_mean": preds_b[i],
            "rul_lower_5":      lo,
            "rul_upper_95":     hi,
        })
    unc_df_d = pd.DataFrame(unc_rows)

    try:
        _, surv_metrics_d = run_survival_risk(
            df=test_df, output_dir=surv_dir_d,
            feature_cols=DEFAULT_FEATURES, horizon=cfg.risk.horizon_cycles,
        )
        surv_df_d = pd.read_csv(surv_dir_d / "survival_risk_predictions.csv")
    except Exception:
        surv_df_d = pd.DataFrame()

    audit_cfg_d = AuditConfig()
    _, audit_metrics_d = run_reliability_audit(
        features_df    = test_df,
        uncertainty_df = unc_df_d,
        survival_df    = surv_df_d,
        anomaly_df     = pd.DataFrame(),
        output_dir     = surv_dir_d,
        audit_cfg      = audit_cfg_d,
    )

    label_counts = audit_metrics_d.get("label_counts", {})
    n_total_d    = audit_metrics_d.get("n_batteries", 1) or 1
    pass_r   = label_counts.get("PASS",    0) / n_total_d
    warn_r   = label_counts.get("WARNING", 0) / n_total_d
    fail_r   = label_counts.get("FAIL",    0) / n_total_d
    severe_r = audit_metrics_d.get("severe_error_catch_rate")

    print(f"   RMSE={rmse_b:.2f}  Coverage={coverage_b*100:.1f}%  "
          f"PASS={pass_r*100:.0f}%  WARN={warn_r*100:.0f}%  FAIL={fail_r*100:.0f}%  "
          f"SevereCatch={f'{severe_r*100:.0f}%' if severe_r is not None else 'N/A'}")

    results.append(AblationResult(
        config="D", description="Full system (XGBoost + uncertainty + risk + audit)",
        rmse=round(rmse_b, 4), mae=round(mae_b, 4),
        coverage=round(coverage_b, 4), mean_interval_width=round(iw_b, 2) if iw_b else None,
        pass_rate=round(pass_r, 4), warning_rate=round(warn_r, 4), fail_rate=round(fail_r, 4),
        severe_catch_rate=round(severe_r, 4) if severe_r is not None else None,
    ))

    # ── Save results ──────────────────────────────────────────────────────────
    rows = [
        {
            "config":              r.config,
            "description":         r.description,
            "rmse":                r.rmse,
            "mae":                 r.mae,
            "coverage":            r.coverage,
            "mean_interval_width": r.mean_interval_width,
            "pass_rate":           r.pass_rate,
            "warning_rate":        r.warning_rate,
            "fail_rate":           r.fail_rate,
            "severe_catch_rate":   r.severe_catch_rate,
        }
        for r in results
    ]
    df_out = pd.DataFrame(rows)

    csv_path  = OUTPUT_DIR / "ablation_results.csv"
    json_path = OUTPUT_DIR / "ablation_results.json"
    df_out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print("=" * 65)
    print("ABLATION SUMMARY")
    print("=" * 65)
    print(df_out.to_string(index=False))
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    return results


if __name__ == "__main__":
    run_ablation()

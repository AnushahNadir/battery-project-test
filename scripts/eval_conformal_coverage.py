"""
Conformal prediction ablation: coverage by temperature group.

Compares three strategies on the same train/cal/test split:

  1. Global split     — one q_hat from all calibration batteries, no stratification
  2. Stratified split — per-group q_hat, standard split conformal (no LOBO)
  3. Stratified LOBO  — per-group q_hat, LOBO for cold batteries  [current method]

The key question: does LOBO improve cold-battery coverage without sacrificing
room/hot coverage?

Outputs
-------
  - Console table (also written to trained_models/conformal_ablation.json)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.log import setup_logging
from src.config import get_config
from src.modeling.conformal import (
    ConformalCalibrator,
    TEMP_GROUPS,
    _get_temp_group,
    _nonconformity_scores,
    select_calibration_batteries,
)
from src.modeling.ml_model import train_ml_model
from src.modeling.run_full_pipeline import (
    _stratified_battery_split,
    _random_battery_split,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global (naive) calibrator — single q_hat, no temperature stratification
# ---------------------------------------------------------------------------

def _fit_global_calibrator(
    cal_df: pd.DataFrame,
    model,
    coverage: float,
    rul_col: str = "RUL",
) -> float:
    """Return a single global q_hat from all calibration batteries."""
    scores = _nonconformity_scores(cal_df, model, rul_col=rul_col)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("No valid calibration scores.")
    alpha = 1.0 - coverage
    n = len(scores)
    level = min(float(np.ceil((n + 1) * (1 - alpha)) / n), 1.0)
    return float(np.quantile(scores, level))


def _coverage_with_global_q(
    test_df: pd.DataFrame,
    model,
    q_hat: float,
    coverage: float,
    rul_col: str = "RUL",
) -> dict:
    """Evaluate coverage of a global q_hat on the test set, broken down by group."""
    rul_col = rul_col if rul_col in test_df.columns else "rul"
    feature_cols = [c for c in model.feature_columns if c in test_df.columns]
    preds = np.asarray(model.predict(test_df[feature_cols]), dtype=float)
    y_true = pd.to_numeric(test_df[rul_col], errors="coerce").to_numpy(dtype=float)

    lo = preds - q_hat
    hi = preds + q_hat
    valid = np.isfinite(y_true) & np.isfinite(preds)
    covered = valid & (y_true >= lo) & (y_true <= hi)
    overall = float(covered[valid].mean()) if valid.any() else float("nan")

    groups = test_df["battery_id"].astype(str).map(_get_temp_group).to_numpy()
    report = {
        "target_coverage": coverage,
        "overall_empirical_coverage": round(overall, 4),
        "q_hat": round(q_hat, 2),
        "per_group": {},
    }
    for grp in TEMP_GROUPS:
        mask = groups == grp
        if not mask.any():
            continue
        gv = valid[mask]
        gc = covered[mask]
        emp = float(gc[gv].mean()) if gv.any() else float("nan")
        report["per_group"][grp] = {
            "n_rows": int(mask.sum()),
            "empirical_coverage": round(emp, 4) if np.isfinite(emp) else None,
            "q_hat": round(q_hat, 2),
            "strategy": "global",
        }
    return report


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def _print_table(results: dict[str, dict], target: float) -> None:
    methods = list(results.keys())
    groups = [g for g in TEMP_GROUPS if any(
        g in r.get("per_group", {}) for r in results.values()
    )]

    col_w = 18
    grp_w = 12

    header = f"{'Method':<{col_w}} | {'Overall':>{grp_w}}"
    for g in groups:
        header += f" | {g.capitalize():>{grp_w}}"
    sep = "-" * len(header)

    logger.info("")
    logger.info("Conformal Coverage Ablation")
    logger.info(f"Target coverage: {target:.0%}")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for name, r in results.items():
        overall = r.get("overall_empirical_coverage")
        ov_str = f"{overall:.1%}" if overall is not None else "N/A"
        if overall is not None and overall < target:
            ov_str += " *"
        row = f"{name:<{col_w}} | {ov_str:>{grp_w}}"
        for g in groups:
            gd = r.get("per_group", {}).get(g, {})
            emp = gd.get("empirical_coverage")
            s = f"{emp:.1%}" if emp is not None else "N/A"
            if emp is not None and emp < target:
                s += " *"
            row += f" | {s:>{grp_w}}"
        logger.info(row)

    logger.info(sep)
    logger.info("* UNDER-COVERED (below target)")
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    cfg = get_config()

    base = ROOT
    data_path = base / cfg.paths.processed_data / "cycle_features_with_rul.csv"
    out_path = base / cfg.paths.trained_models / "conformal_ablation.json"

    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    batteries = sorted(df["battery_id"].dropna().astype(str).unique())

    # --- same split logic as run_full_pipeline ---
    if cfg.split.stratify_by == "temperature_group":
        train_batteries, test_batteries = _stratified_battery_split(
            batteries=batteries,
            train_ratio=cfg.split.train_ratio,
            random_seed=cfg.split.random_seed,
        )
    else:
        train_batteries, test_batteries = _random_battery_split(
            batteries=batteries,
            train_ratio=cfg.split.train_ratio,
            random_seed=cfg.split.random_seed,
        )

    cal_batteries, actual_train_batteries = select_calibration_batteries(
        train_batteries=train_batteries,
        n_cal=cfg.split.n_cal_batteries,
        random_seed=cfg.split.random_seed,
    )

    train_df = df[df["battery_id"].astype(str).isin(actual_train_batteries)].copy()
    cal_df   = df[df["battery_id"].astype(str).isin(cal_batteries)].copy()
    test_df  = df[df["battery_id"].astype(str).isin(test_batteries)].copy()

    logger.info(f"Train: {len(actual_train_batteries)} batteries  "
                f"Cal: {len(cal_batteries)} batteries  "
                f"Test: {len(test_batteries)} batteries")
    logger.info(f"Test batteries: {test_batteries}")

    # --- train one shared XGBoost model ---
    logger.info("Training XGBoost model...")
    trained_models_dir = base / cfg.paths.trained_models
    trained_models_dir.mkdir(parents=True, exist_ok=True)
    ml_model, ml_metrics = train_ml_model(
        df, actual_train_batteries, test_batteries, trained_models_dir
    )
    logger.info(f"Model RMSE on test: {ml_metrics.rmse:.2f}")

    coverage = cfg.conformal.coverage

    # --- Method 1: Global split ---
    logger.info("Fitting global (naive) calibrator...")
    global_q = _fit_global_calibrator(cal_df, ml_model, coverage)
    logger.info(f"  global q_hat = {global_q:.2f}")
    result_global = _coverage_with_global_q(test_df, ml_model, global_q, coverage)

    # --- Method 2: Stratified split (no LOBO) ---
    logger.info("Fitting stratified-split calibrator (no LOBO)...")
    cal_split = ConformalCalibrator(coverage=coverage)
    cal_split.fit(cal_df, ml_model, lobo_groups_override=set())
    result_split = cal_split.coverage_report(test_df, ml_model)

    # --- Method 3: Stratified LOBO (current method) ---
    logger.info("Fitting stratified-LOBO calibrator...")
    # For LOBO we pass the full training data so it can refit per battery.
    train_df_for_lobo = df[df["battery_id"].astype(str).isin(train_batteries)].copy()
    cal_lobo = ConformalCalibrator(coverage=coverage)
    cal_lobo.fit(
        cal_df,
        ml_model,
        train_df=train_df_for_lobo,
        actual_train_batteries=actual_train_batteries,
    )
    result_lobo = cal_lobo.coverage_report(test_df, ml_model)

    # --- Print comparison table ---
    results = {
        "Global split":      result_global,
        "Stratified split":  result_split,
        "Stratified LOBO":   result_lobo,
    }
    _print_table(results, coverage)

    # --- Save ---
    output = {
        "target_coverage": coverage,
        "test_batteries": test_batteries,
        "ml_rmse": round(ml_metrics.rmse, 4),
        "methods": {k: v for k, v in results.items()},
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

"""
Drift-vs-coverage linkage analysis for the battery RUL paper.

For each test battery:
  - Computes per-battery PSI (max and mean across monitored features)
    comparing the test battery's feature distribution against the training distribution.
  - Computes per-battery empirical conformal coverage (fraction of cycles where
    true RUL falls within the [rul_lower_5, rul_upper_95] interval).
  - Reports per-battery RMSE from trained_models/model_metrics.json.

Output:
  data/processed/modeling/drift_coverage_linkage.json

Usage:
  python scripts/drift_coverage_linkage.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.log import setup_logging

logger = logging.getLogger(__name__)

PSI_FEATURES = [
    "temp_mean", "temp_max", "v_mean", "i_mean", "duration_s",
    "capacity", "ah_est", "energy_j",
]
PSI_BINS = 10
PSI_EPSILON = 1e-6


def _psi_score(train_vals: np.ndarray, test_vals: np.ndarray, n_bins: int = PSI_BINS) -> float:
    """Population Stability Index: quantifies distribution shift train → test."""
    train_vals = train_vals[np.isfinite(train_vals)]
    test_vals = test_vals[np.isfinite(test_vals)]
    if len(train_vals) < 2 or len(test_vals) < 2:
        return float("nan")
    combined = np.concatenate([train_vals, test_vals])
    edges = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    train_counts, _ = np.histogram(train_vals, bins=edges)
    test_counts, _ = np.histogram(test_vals, bins=edges)

    train_frac = train_counts / max(train_counts.sum(), 1) + PSI_EPSILON
    test_frac = test_counts / max(test_counts.sum(), 1) + PSI_EPSILON

    psi = float(np.sum((test_frac - train_frac) * np.log(test_frac / train_frac)))
    return round(psi, 6)


def _temp_group(battery_id: str, df: pd.DataFrame) -> str:
    rows = df[df["battery_id"].astype(str) == str(battery_id)]
    if rows.empty or "temp_group" not in rows.columns:
        return "unknown"
    return str(rows["temp_group"].iloc[0])


def main() -> None:
    setup_logging()

    data_path = Path("data/processed/cycle_features_with_rul.csv")
    estimates_path = Path("data/processed/modeling/uncertainty_estimates.json")
    model_metrics_path = Path("trained_models/model_metrics.json")
    output_path = Path("data/processed/modeling/drift_coverage_linkage.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not estimates_path.exists():
        raise FileNotFoundError(f"Uncertainty estimates not found: {estimates_path}")

    df_all = pd.read_csv(data_path)
    estimates = json.loads(estimates_path.read_text(encoding="utf-8"))

    est_df = pd.DataFrame(estimates)
    est_df["battery_id"] = est_df["battery_id"].astype(str)

    per_battery_rmse: Dict[str, float] = {}
    if model_metrics_path.exists():
        metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
        per_battery_rmse = {
            str(k): float(v)
            for k, v in metrics.get("per_battery_rmse", {}).items()
        }

    in_test_col = est_df.get("in_test_set", pd.Series(False, index=est_df.index))
    test_estimates = est_df[in_test_col == True].copy()
    if test_estimates.empty:
        logger.warning("No records with in_test_set=True found; using all estimate records.")
        test_estimates = est_df.copy()

    test_batteries = sorted(test_estimates["battery_id"].unique())
    logger.info(f"Test batteries: {test_batteries}")

    all_batteries = set(df_all["battery_id"].astype(str).unique())
    train_batteries = all_batteries - set(test_batteries)
    train_df = df_all[df_all["battery_id"].astype(str).isin(train_batteries)]

    df_all["battery_id"] = df_all["battery_id"].astype(str)
    rul_lookup = df_all[["battery_id", "cycle_index", "RUL"]].copy()
    rul_lookup["cycle_index"] = pd.to_numeric(rul_lookup["cycle_index"], errors="coerce")

    if "cycle_index" in test_estimates.columns:
        test_estimates["cycle_index"] = pd.to_numeric(
            test_estimates["cycle_index"], errors="coerce"
        )
        test_estimates = test_estimates.merge(
            rul_lookup, on=["battery_id", "cycle_index"], how="left"
        )

    records: List[Dict[str, Any]] = []
    for bat in test_batteries:
        bat_est = test_estimates[test_estimates["battery_id"] == bat].copy()
        bat_cycles = df_all[df_all["battery_id"] == bat]
        temp_grp = _temp_group(bat, df_all)

        coverage: Optional[float] = None
        if (
            "RUL" in bat_est.columns
            and "rul_lower_5" in bat_est.columns
            and "rul_upper_95" in bat_est.columns
        ):
            y_true = pd.to_numeric(bat_est["RUL"], errors="coerce").to_numpy(float)
            lo = pd.to_numeric(bat_est["rul_lower_5"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(bat_est["rul_upper_95"], errors="coerce").to_numpy(float)
            valid = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
            if valid.any():
                covered = ((y_true[valid] >= lo[valid]) & (y_true[valid] <= hi[valid])).mean()
                coverage = round(float(covered), 4)

        psi_by_feature: Dict[str, float] = {}
        for feat in PSI_FEATURES:
            if feat not in train_df.columns or feat not in bat_cycles.columns:
                continue
            train_vals = pd.to_numeric(train_df[feat], errors="coerce").dropna().to_numpy(float)
            test_vals = pd.to_numeric(bat_cycles[feat], errors="coerce").dropna().to_numpy(float)
            psi_by_feature[feat] = _psi_score(train_vals, test_vals)

        finite_psi = [v for v in psi_by_feature.values() if np.isfinite(v)]
        psi_max = round(float(max(finite_psi)), 4) if finite_psi else None
        psi_mean = round(float(np.mean(finite_psi)), 4) if finite_psi else None

        rmse = per_battery_rmse.get(bat)

        rec: Dict[str, Any] = {
            "battery_id": bat,
            "temp_group": temp_grp,
            "n_test_cycles": int(len(bat_est)),
            "empirical_coverage": coverage,
            "rmse": round(rmse, 4) if rmse is not None else None,
            "psi_max": psi_max,
            "psi_mean": psi_mean,
            "psi_by_feature": psi_by_feature,
        }
        records.append(rec)
        rmse_str = f"{rmse:.2f}" if rmse is not None else "N/A"
        logger.info(
            f"  {bat} ({temp_grp}): coverage={coverage}, rmse={rmse_str}, "
            f"psi_max={psi_max}, psi_mean={psi_mean}"
        )

    payload: Dict[str, Any] = {
        "description": (
            "Per-battery PSI (training-distribution shift), empirical conformal coverage, "
            "and RMSE for the test batteries. Used to link feature drift to under-coverage "
            "for the paper's drift-vs-coverage figure."
        ),
        "n_train_batteries": len(train_batteries),
        "n_test_batteries": len(test_batteries),
        "psi_features": PSI_FEATURES,
        "coverage_interval": "rul_lower_5 to rul_upper_95 (90% conformal)",
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"\nDrift-coverage linkage written to {output_path}")


if __name__ == "__main__":
    main()

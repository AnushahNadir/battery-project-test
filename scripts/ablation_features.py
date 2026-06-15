"""
Feature ablation study for the battery RUL paper.

Runs three configurations across multiple seeds:
  full            — all features (baseline)
  no_cycle_index  — cycle_index removed (leakage check)
  no_capacity     — capacity removed (target-adjacent feature check)

Each configuration uses the same stratified battery split and anomaly filter
as the main pipeline. Results are written to:
  trained_models/ablation_results.json

Usage:
  python scripts/ablation_features.py
  python scripts/ablation_features.py --seeds 42,7,11,21,99
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.log import setup_logging
from src.modeling.ml_model import FEATURE_COLUMNS, XGBoostRULModel
from src.modeling.run_full_pipeline import (
    _apply_pretrain_anomaly_flags,
    _compute_uncertainty_weights,
    _fit_pretrain_anomaly_thresholds,
    _stratified_battery_split,
)
from src.modeling.statistical_baseline import StatisticalBaselineModel

logger = logging.getLogger(__name__)

SEEDS = [42, 7, 11, 21, 99]

CONFIGS: Dict[str, List[str]] = {
    "full": list(FEATURE_COLUMNS),
    "no_cycle_index": [f for f in FEATURE_COLUMNS if f != "cycle_index"],
    "no_capacity": [f for f in FEATURE_COLUMNS if f != "capacity"],
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _run_config_seed(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    train_ratio: float = 0.8,
    capacity_floor_factor: float = 0.5,
) -> Dict[str, Any]:
    _set_seed(seed)
    batteries = sorted(df["battery_id"].dropna().astype(str).unique())
    train_bats, test_bats = _stratified_battery_split(
        batteries=batteries, train_ratio=train_ratio, random_seed=seed
    )

    train_raw = df[df["battery_id"].astype(str).isin(train_bats)].copy()
    test_df = df[df["battery_id"].astype(str).isin(test_bats)].copy()

    thresholds, global_thr = _fit_pretrain_anomaly_thresholds(
        train_df=train_raw,
        capacity_col="capacity",
        capacity_floor_factor=capacity_floor_factor,
    )
    train_flagged = _apply_pretrain_anomaly_flags(
        train_raw, thresholds, global_thr, capacity_col="capacity"
    )
    train_df = train_flagged[~train_flagged["pretrain_anomaly_flag"]].copy()

    available = [c for c in feature_cols if c in train_df.columns]

    ml = XGBoostRULModel(feature_columns=available)
    ml.fit(train_df, quantiles=True)
    ml_metrics = ml.evaluate(test_df)

    stat = StatisticalBaselineModel()
    stat.fit(train_df)
    stat_metrics = stat.evaluate(test_df)

    return {
        "seed": seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "xgb_rmse": round(float(ml_metrics.rmse), 4),
        "xgb_mae": round(float(ml_metrics.mae), 4),
        "stat_rmse": round(float(stat_metrics.rmse), 4),
        "feature_cols": available,
        "n_features": len(available),
    }


def _summary(runs: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [r[key] for r in runs if key in r and np.isfinite(r[key])]
    if not vals:
        return {}
    arr = np.array(vals, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=0)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "n": len(arr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path,
        default=Path("data/processed/cycle_features_with_rul.csv"),
    )
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--capacity-floor-factor", type=float, default=0.5)
    parser.add_argument(
        "--output-json", type=Path,
        default=Path("trained_models/ablation_results.json"),
    )
    args = parser.parse_args()
    setup_logging()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    df = pd.read_csv(args.data_path)

    results: Dict[str, Any] = {}
    for cfg_name, feat_cols in CONFIGS.items():
        logger.info(f"\n=== Config: {cfg_name} ({len(feat_cols)} features) ===")
        runs: List[Dict[str, Any]] = []
        for seed in seeds:
            logger.info(f"  Seed {seed}...")
            try:
                run = _run_config_seed(
                    df=df, feature_cols=feat_cols, seed=seed,
                    train_ratio=args.train_ratio,
                    capacity_floor_factor=args.capacity_floor_factor,
                )
                runs.append(run)
                logger.info(f"    xgb_rmse={run['xgb_rmse']:.2f}  stat_rmse={run['stat_rmse']:.2f}")
            except Exception as exc:
                logger.warning(f"    Seed {seed} failed: {exc}")
        results[cfg_name] = {
            "features": feat_cols,
            "n_features": len(feat_cols),
            "runs": runs,
            "summary": {
                "xgb_rmse": _summary(runs, "xgb_rmse"),
                "stat_rmse": _summary(runs, "stat_rmse"),
            },
        }
        if runs:
            xgb_sum = results[cfg_name]["summary"]["xgb_rmse"]
            logger.info(
                f"  {cfg_name}: XGB RMSE = {xgb_sum['mean']:.2f} ± {xgb_sum['std']:.2f}"
            )

    if "full" in results and results["full"]["summary"].get("xgb_rmse"):
        full_mean = results["full"]["summary"]["xgb_rmse"]["mean"]
        for cfg_name in ("no_cycle_index", "no_capacity"):
            if cfg_name in results and results[cfg_name]["summary"].get("xgb_rmse"):
                ablated_mean = results[cfg_name]["summary"]["xgb_rmse"]["mean"]
                delta = round(ablated_mean - full_mean, 4)
                results[cfg_name]["delta_vs_full_xgb_rmse"] = delta
                logger.info(f"  {cfg_name}: ΔRMSE vs full = {delta:+.2f} cycles")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"\nAblation results written to {args.output_json}")


if __name__ == "__main__":
    main()

"""
Benchmark model performance across multiple battery-level split seeds.

This script reuses the same split/filter logic from the full pipeline:
- stratified battery split by temperature group
- pre-train anomaly threshold fit on train only
- anomaly removal from train only (test is flagged, not dropped)
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

# Ensure project root is on path when invoked as a script.
sys.path.append(str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

from src.log import setup_logging
from src.modeling.dl_sequence_model import TemporalSequenceModel
from src.modeling.ml_model import XGBoostRULModel
from src.modeling.run_full_pipeline import (
    _apply_pretrain_anomaly_flags,
    _fit_pretrain_anomaly_thresholds,
    _group_counts,
    _stratified_battery_split,
)
from src.modeling.statistical_baseline import StatisticalBaselineModel


def _parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("No valid seeds were provided.")
    return seeds


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _prepare_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    capacity_floor_factor: float,
) -> Dict[str, Any]:
    batteries = sorted(df["battery_id"].dropna().astype(str).unique())
    train_batteries, test_batteries = _stratified_battery_split(
        batteries=batteries,
        train_ratio=train_ratio,
        random_seed=seed,
    )

    train_df_raw = df[df["battery_id"].astype(str).isin(train_batteries)].copy()
    test_df = df[df["battery_id"].astype(str).isin(test_batteries)].copy()

    thresholds, global_threshold = _fit_pretrain_anomaly_thresholds(
        train_df=train_df_raw,
        capacity_col="capacity",
        capacity_floor_factor=capacity_floor_factor,
    )
    train_df_flagged = _apply_pretrain_anomaly_flags(
        train_df_raw,
        thresholds,
        global_threshold,
        capacity_col="capacity",
    )
    test_df = _apply_pretrain_anomaly_flags(
        test_df,
        thresholds,
        global_threshold,
        capacity_col="capacity",
    )

    dropped_train = int(train_df_flagged["pretrain_anomaly_flag"].sum())
    train_df = train_df_flagged[~train_df_flagged["pretrain_anomaly_flag"]].copy()
    model_df = pd.concat([train_df, test_df], ignore_index=True)

    return {
        "seed": seed,
        "train_batteries": train_batteries,
        "test_batteries": test_batteries,
        "train_temp_mix": _group_counts(train_batteries),
        "test_temp_mix": _group_counts(test_batteries),
        "train_rows_raw": int(len(train_df_raw)),
        "train_rows_clean": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_rows_removed": dropped_train,
        "test_rows_flagged": int(test_df["pretrain_anomaly_flag"].sum()),
        "df_model": model_df,
    }


def _rmse_summary(runs: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals: List[float] = []
    for run in runs:
        v = run.get(key)
        if v is None:
            continue
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _run_seed(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    capacity_floor_factor: float,
    include_dl: bool,
    dl_epochs: int,
    dl_patience: int,
    dl_hidden: int,
    dl_seq_len: int,
    dl_batch_size: int,
    dl_lr: float,
    dl_dropout: float,
    dl_weight_decay: float,
    dl_backend: str,
) -> Dict[str, Any]:
    _set_seed(seed)
    split = _prepare_split(
        df=df,
        seed=seed,
        train_ratio=train_ratio,
        capacity_floor_factor=capacity_floor_factor,
    )
    model_df = split.pop("df_model")
    train_bats = split["train_batteries"]
    test_bats = split["test_batteries"]
    train_df = model_df[model_df["battery_id"].astype(str).isin(train_bats)].copy()
    test_df = model_df[model_df["battery_id"].astype(str).isin(test_bats)].copy()

    result: Dict[str, Any] = {"seed": seed, **split}

    stat_model = StatisticalBaselineModel()
    stat_model.fit(train_df)
    stat_metrics = stat_model.evaluate(test_df)
    result["stat_rmse"] = float(stat_metrics.rmse)
    result["stat_mae"] = float(stat_metrics.mae)

    ml_model = XGBoostRULModel()
    ml_model.fit(train_df, quantiles=True)
    ml_metrics = ml_model.evaluate(test_df)
    result["xgb_rmse"] = float(ml_metrics.rmse)
    result["xgb_mae"] = float(ml_metrics.mae)

    if include_dl:
        dl_model = TemporalSequenceModel(
            sequence_length=dl_seq_len,
            hidden_channels=dl_hidden,
            learning_rate=dl_lr,
            batch_size=dl_batch_size,
            epochs=dl_epochs,
            patience=dl_patience,
            dropout=dl_dropout,
            weight_decay=dl_weight_decay,
            force_backend=dl_backend,
            use_log_target=False,
            huber_beta=3.0,
            grad_clip=1.0,
            lr_factor=0.5,
            lr_patience=4,
        )
        dl_model.fit(train_df)
        dl_metrics = dl_model.evaluate(test_df)
        result["dl_rmse"] = float(dl_metrics.rmse)
        result["dl_mae"] = float(dl_metrics.mae)
        result["dl_backend"] = dl_metrics.backend

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed benchmark for battery RUL models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/cycle_features_with_rul.csv"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,7,11,21,99",
        help="Comma-separated seeds (example: 42,7,11,21,99).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--capacity-floor-factor", type=float, default=0.5)
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--dl-epochs", type=int, default=60)
    parser.add_argument("--dl-patience", type=int, default=10)
    parser.add_argument("--dl-hidden", type=int, default=32)
    parser.add_argument("--dl-seq-len", type=int, default=8)
    parser.add_argument("--dl-batch-size", type=int, default=64)
    parser.add_argument("--dl-lr", type=float, default=1.0e-3)
    parser.add_argument("--dl-dropout", type=float, default=0.1)
    parser.add_argument("--dl-weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--dl-backend",
        type=str,
        choices=["auto", "tcn", "mlp"],
        default="auto",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("trained_models/multi_seed_benchmark.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("trained_models/multi_seed_benchmark.csv"),
    )
    args = parser.parse_args()
    setup_logging()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    seeds = _parse_seeds(args.seeds)
    include_dl = not args.skip_dl

    df = pd.read_csv(args.data_path)
    if "battery_id" not in df.columns:
        raise ValueError("Input dataframe must contain 'battery_id'.")

    runs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for seed in seeds:
        logger.info(f"\n=== Seed {seed} ===")
        try:
            run = _run_seed(
                df=df,
                seed=seed,
                train_ratio=args.train_ratio,
                capacity_floor_factor=args.capacity_floor_factor,
                include_dl=include_dl,
                dl_epochs=args.dl_epochs,
                dl_patience=args.dl_patience,
                dl_hidden=args.dl_hidden,
                dl_seq_len=args.dl_seq_len,
                dl_batch_size=args.dl_batch_size,
                dl_lr=args.dl_lr,
                dl_dropout=args.dl_dropout,
                dl_weight_decay=args.dl_weight_decay,
                dl_backend=args.dl_backend,
            )
            runs.append(run)
            logger.info(
                f"RMSE | stat={run['stat_rmse']:.2f}, xgb={run['xgb_rmse']:.2f}"
                + (f", dl={run['dl_rmse']:.2f}" if include_dl and "dl_rmse" in run else "")
            )
        except Exception as exc:
            err = {"seed": seed, "error": str(exc)}
            errors.append(err)
            logger.info(f"Seed {seed} failed: {exc}")

    summary = {
        "statistical_rmse": _rmse_summary(runs, "stat_rmse"),
        "xgboost_rmse": _rmse_summary(runs, "xgb_rmse"),
    }
    if include_dl:
        summary["dl_rmse"] = _rmse_summary(runs, "dl_rmse")

    payload = {
        "config": {
            "data_path": str(args.data_path),
            "seeds": seeds,
            "train_ratio": args.train_ratio,
            "capacity_floor_factor": args.capacity_floor_factor,
            "include_dl": include_dl,
            "dl": {
                "epochs": args.dl_epochs,
                "patience": args.dl_patience,
                "hidden_channels": args.dl_hidden,
                "sequence_length": args.dl_seq_len,
                "batch_size": args.dl_batch_size,
                "learning_rate": args.dl_lr,
                "dropout": args.dl_dropout,
                "weight_decay": args.dl_weight_decay,
                "backend": args.dl_backend,
            },
        },
        "runs": runs,
        "summary": summary,
        "errors": errors,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if runs:
        pd.DataFrame(runs).to_csv(args.output_csv, index=False)

    logger.info("\n=== Aggregate RMSE ===")
    for k, v in summary.items():
        if v.get("n", 0) == 0:
            logger.info(f"{k}: no successful runs")
        else:
            logger.info(f"{k}: mean={v['mean']:.2f}, std={v['std']:.2f}, min={v['min']:.2f}, max={v['max']:.2f}, n={v['n']}")
    logger.info(f"Saved JSON: {args.output_json}")
    if runs:
        logger.info(f"Saved CSV:  {args.output_csv}")
    if errors:
        logger.info(f"Failed seeds: {[e['seed'] for e in errors]}")


if __name__ == "__main__":
    main()

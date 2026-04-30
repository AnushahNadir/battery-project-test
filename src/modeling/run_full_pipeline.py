# src/modeling/run_full_pipeline.py
"""
Orchestration script for the Battery AI Co-Scientist (Stages 3-6).

Stages:
1. Stage 3: Degradation Modeling (Statistical + ML)
2. Stage 4: Uncertainty Quantification
3. Stage 4.2: Survival/Hazard Risk Estimation
4. Stage 4.5: Anomaly Detection
5. Stage 5: Reasoning & Counterfactuals
6. Stage 6: Supervisor Review

Input:
- data/processed/cycle_features_with_rul.csv

Outputs:
- trained_models/
- data/processed/modeling/
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path when invoked as a script.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.log import setup_logging
from src.modeling.anomaly_detection import run_anomaly_detection

logger = logging.getLogger(__name__)
from src.modeling.ml_model import (
    FEATURE_COLUMNS,
    evaluate_groupkfold_cv,
    train_ml_model,
)
from src.modeling.dl_sequence_model import train_dl_sequence_model
from src.modeling.reasoning import run_reasoning_analysis
from src.modeling.statistical_baseline import train_statistical_baseline
from src.modeling.supervisor_review import SupervisorReviewer
from src.modeling.uncertainty import run_uncertainty_analysis
from src.modeling.conformal import AdaptiveConformalCalibrator, BootstrapCalibrator, ConformalCalibrator, select_calibration_batteries
from src.analysis.survival_risk import DEFAULT_FEATURES, run_survival_risk
from src.pipeline.drift import DriftMonitor
from src.config import get_config


def _battery_number(battery_id: str) -> int | None:
    bid = str(battery_id).strip().upper()
    if not (bid.startswith("B") and bid[1:].isdigit()):
        return None
    return int(bid[1:])


def _get_temp_group(battery_id: str) -> str:
    """
    Infer temperature group from NASA battery ID blocks.
    Fallback is room temperature if an ID is unknown.
    """
    num = _battery_number(battery_id)
    if num is None:
        return "room"
    if 41 <= num <= 56:
        return "cold"
    if 29 <= num <= 32 or 38 <= num <= 40:
        return "hot"
    return "room"


def _stratified_battery_split(
    batteries: list[str],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(random_seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for b in sorted(set(str(x) for x in batteries)):
        groups[_get_temp_group(b)].append(b)

    train_ids: list[str] = []
    test_ids: list[str] = []

    for group_name in sorted(groups):
        group_bats = groups[group_name][:]
        rng.shuffle(group_bats)
        n = len(group_bats)

        if n <= 1:
            train_ids.extend(group_bats)
            continue

        n_train = int(round(train_ratio * n))
        n_train = max(1, min(n_train, n - 1))

        train_ids.extend(group_bats[:n_train])
        test_ids.extend(group_bats[n_train:])

    if not test_ids and len(train_ids) > 1:
        test_ids.append(train_ids.pop())

    return sorted(train_ids), sorted(test_ids)


def _random_battery_split(
    batteries: list[str],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    rng = random.Random(random_seed)
    ids = sorted(set(str(x) for x in batteries))
    rng.shuffle(ids)
    if len(ids) <= 1:
        return ids, []
    n_train = int(round(train_ratio * len(ids)))
    n_train = max(1, min(n_train, len(ids) - 1))
    return sorted(ids[:n_train]), sorted(ids[n_train:])


def _is_flagged_battery(battery_id: str, flagged_prefixes: list[str]) -> bool:
    bid = str(battery_id)
    return any(bid.startswith(prefix) for prefix in flagged_prefixes)


def _fit_pretrain_anomaly_thresholds(
    train_df: pd.DataFrame,
    capacity_col: str = "capacity",
    capacity_floor_factor: float = 0.5,
    flagged_prefixes: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    if capacity_col not in train_df.columns:
        return {}, 0.0

    flagged_prefixes = flagged_prefixes or []

    work = train_df[["battery_id", capacity_col]].copy()
    work[capacity_col] = pd.to_numeric(work[capacity_col], errors="coerce")

    thresholds: dict[str, float] = {}
    fitted_values: list[float] = []

    for bid, group in work.groupby("battery_id", sort=False):
        bid_str = str(bid)
        if not _is_flagged_battery(bid_str, flagged_prefixes):
            continue
        med = float(np.nanmedian(group[capacity_col].values))
        if not np.isfinite(med) or med <= 0:
            continue
        thr = capacity_floor_factor * med
        thresholds[bid_str] = float(thr)
        fitted_values.append(float(thr))

    if fitted_values:
        global_thr = float(np.nanmedian(np.asarray(fitted_values)))
    else:
        global_med = float(np.nanmedian(work[capacity_col].values))
        if not np.isfinite(global_med) or global_med <= 0:
            global_med = 1.0
        global_thr = float(capacity_floor_factor * global_med)

    return thresholds, global_thr


def _apply_pretrain_anomaly_flags(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    global_threshold: float,
    capacity_col: str = "capacity",
    flagged_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if capacity_col not in out.columns:
        out["pretrain_anomaly_flag"] = False
        return out

    flagged_prefixes = flagged_prefixes or []
    cap = pd.to_numeric(out[capacity_col], errors="coerce")
    bid = out["battery_id"].astype(str)
    thr = bid.map(thresholds).fillna(global_threshold)
    flag = cap.notna() & (cap < thr.astype(float))

    # Only apply pre-training anomaly logic to known problematic batteries.
    # If no prefixes are configured the filter is intentionally disabled and
    # every row gets flag=False — no rows are removed.
    if not flagged_prefixes:
        out["pretrain_anomaly_flag"] = False
        return out
    flag &= bid.apply(lambda x: _is_flagged_battery(x, flagged_prefixes))

    out["pretrain_anomaly_flag"] = flag.astype(bool)
    return out


def _group_counts(ids: list[str]) -> dict[str, int]:
    counts = {"room": 0, "hot": 0, "cold": 0}
    for b in ids:
        g = _get_temp_group(b)
        counts[g] = counts.get(g, 0) + 1
    return counts


def _compute_uncertainty_weights(
    stat_rmse: float,
    ml_rmse: float,
    dl_rmse: float | None = None,
    exclusion_ratio: float = 1.5,
) -> dict[str, float]:
    """
    Compute robust ensemble weights from RMSE.

    - Drops any model with RMSE > exclusion_ratio * best_rmse.
    - Uses inverse-RMSE weighting over remaining models.
    - Guarantees at least ML contributes.
    """
    rmse_map: dict[str, float] = {
        "stat": float(stat_rmse),
        "ml": float(ml_rmse),
    }
    if dl_rmse is not None:
        rmse_map["dl"] = float(dl_rmse)

    finite = {k: v for k, v in rmse_map.items() if np.isfinite(v) and v > 0}
    if not finite:
        return {"stat": 0.0, "ml": 1.0, "dl": 0.0}

    best = min(finite.values())
    keep = {k: v for k, v in finite.items() if v <= exclusion_ratio * best}
    if "ml" not in keep and "ml" in finite:
        keep["ml"] = finite["ml"]

    inv = {k: 1.0 / v for k, v in keep.items()}
    den = sum(inv.values())
    if den <= 0:
        return {"stat": 0.0, "ml": 1.0, "dl": 0.0}

    out = {
        "stat": float(inv.get("stat", 0.0) / den),
        "ml": float(inv.get("ml", 0.0) / den),
        "dl": float(inv.get("dl", 0.0) / den),
    }
    return out


def _write_manifest(
    output_dir: Path,
    ml_rmse: float,
    stat_rmse: float,
    n_test_batteries: int,
) -> None:
    """Write manifest.json so it's always clear which run produced these outputs."""
    manifest: dict = {
        "completed_at": datetime.datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "metrics": {
            "ml_rmse": round(ml_rmse, 4),
            "stat_rmse": round(stat_rmse, 4),
            "n_test_batteries": n_test_batteries,
        },
    }
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        manifest["git_hash"] = git_hash
    except Exception:
        pass
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest written to {manifest_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    cfg = get_config()
    config_path = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
    logger.info(f"[Config] Loaded from: {config_path.resolve()}")
    logger.info(f"[Config] {cfg.summary()}")

    data_path = base_dir / cfg.paths.processed_data / "cycle_features_with_rul.csv"
    trained_models_dir = base_dir / cfg.paths.trained_models
    output_dir = base_dir / cfg.paths.processed_data / "modeling"

    trained_models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir=output_dir)

    logger.info(f"Loading data from {data_path}...")

    if not data_path.exists():
        logger.critical("Preprocessed dataset not found. Run Stage 1-2 preprocessing first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    if "battery_id" not in df.columns:
        logger.critical("Missing required column 'battery_id'.")
        sys.exit(1)

    batteries = sorted(df["battery_id"].dropna().astype(str).unique())
    n_total = len(batteries)
    if n_total < 2:
        logger.critical("Need at least 2 batteries for train/test split.")
        sys.exit(1)

    split_seed = cfg.split.random_seed
    if cfg.split.stratify_by == "temperature_group":
        train_batteries, test_batteries = _stratified_battery_split(
            batteries=batteries,
            train_ratio=cfg.split.train_ratio,
            random_seed=split_seed,
        )
        split_strategy = "stratified_battery_split"
    else:
        train_batteries, test_batteries = _random_battery_split(
            batteries=batteries,
            train_ratio=cfg.split.train_ratio,
            random_seed=split_seed,
        )
        split_strategy = "random_battery_split"

    cal_batteries, actual_train_batteries = select_calibration_batteries(
        train_batteries=train_batteries,
        n_cal=cfg.split.n_cal_batteries,
        random_seed=split_seed,
    )

    overlap = set(train_batteries) & set(test_batteries)
    if overlap:
        logger.critical(f"Battery leakage detected: {sorted(overlap)}")
        sys.exit(1)
    cal_overlap = set(actual_train_batteries) & set(cal_batteries)
    if cal_overlap:
        logger.critical(f"Calibration/train overlap detected: {sorted(cal_overlap)}")
        sys.exit(1)
    if (set(actual_train_batteries) | set(cal_batteries)) != set(train_batteries):
        logger.critical("Calibration split does not partition training batteries correctly.")
        sys.exit(1)

    logger.info("[Split]")
    logger.info(f"  Total batteries: {n_total}")
    logger.info(f"  Train batteries: {len(train_batteries)}")
    logger.info(f"  Actual train:    {len(actual_train_batteries)}")
    logger.info(f"  Calibration:     {len(cal_batteries)}")
    logger.info(f"  Test batteries:  {len(test_batteries)}")
    logger.info(f"  Split seed:      {split_seed}")
    logger.info(f"  Train temp mix:  {_group_counts(train_batteries)}")
    logger.info(f"  Actual train mix:{_group_counts(actual_train_batteries)}")
    logger.info(f"  Cal temp mix:    {_group_counts(cal_batteries)}")
    logger.info(f"  Test temp mix:   {_group_counts(test_batteries)}")

    train_df_raw = df[df["battery_id"].astype(str).isin(actual_train_batteries)].copy()
    cal_df_raw = df[df["battery_id"].astype(str).isin(cal_batteries)].copy()
    test_df = df[df["battery_id"].astype(str).isin(test_batteries)].copy()

    if not cfg.anomaly.flagged_battery_prefixes:
        logger.warning(
            "anomaly.flagged_battery_prefixes is empty — "
            "pre-training anomaly filter is disabled. No rows will be removed."
        )

    thresholds, global_threshold = _fit_pretrain_anomaly_thresholds(
        train_df=train_df_raw,
        capacity_col="capacity",
        capacity_floor_factor=cfg.anomaly.capacity_floor_factor,
        flagged_prefixes=cfg.anomaly.flagged_battery_prefixes,
    )
    train_df_flagged = _apply_pretrain_anomaly_flags(
        train_df_raw,
        thresholds,
        global_threshold,
        capacity_col="capacity",
        flagged_prefixes=cfg.anomaly.flagged_battery_prefixes,
    )
    cal_df = _apply_pretrain_anomaly_flags(
        cal_df_raw,
        thresholds,
        global_threshold,
        capacity_col="capacity",
        flagged_prefixes=cfg.anomaly.flagged_battery_prefixes,
    )
    test_df = _apply_pretrain_anomaly_flags(
        test_df,
        thresholds,
        global_threshold,
        capacity_col="capacity",
        flagged_prefixes=cfg.anomaly.flagged_battery_prefixes,
    )

    dropped_train_rows = int(train_df_flagged["pretrain_anomaly_flag"].sum())
    train_df = train_df_flagged[~train_df_flagged["pretrain_anomaly_flag"]].copy()
    logger.info(
        f"  Pre-train anomaly rows removed from train: {dropped_train_rows}/{len(train_df_raw)} "
        f"({(100*dropped_train_rows/len(train_df_raw)) if len(train_df_raw) else 0:.1f}%)"
    )
    logger.info(
        f"  Pre-train anomaly rows flagged in test: {int(test_df['pretrain_anomaly_flag'].sum())}/{len(test_df)}"
    )
    logger.info(
        f"  Pre-train anomaly rows flagged in calibration: "
        f"{int(cal_df['pretrain_anomaly_flag'].sum())}/{len(cal_df)}"
    )

    # Keep calibration/test rows untouched; only clean actual training rows.
    df_model = pd.concat([train_df, cal_df, test_df], ignore_index=True)

    logger.info("=" * 50)
    logger.info("STAGE 3: MODELING")
    logger.info("=" * 50)

    stat_model, stat_metrics = train_statistical_baseline(
        df_model,
        actual_train_batteries,
        test_batteries,
        trained_models_dir,
    )
    ml_model, ml_metrics = train_ml_model(
        df_model,
        actual_train_batteries,
        test_batteries,
        trained_models_dir,
    )
    dl_model, dl_metrics = train_dl_sequence_model(
        df_model,
        actual_train_batteries,
        test_batteries,
        trained_models_dir,
        sequence_length=cfg.dl_model.sequence_length,
        hidden_channels=cfg.dl_model.hidden_channels,
        learning_rate=cfg.dl_model.learning_rate,
        batch_size=cfg.dl_model.batch_size,
        epochs=cfg.dl_model.epochs,
        patience=cfg.dl_model.patience,
        dropout=cfg.dl_model.dropout,
        weight_decay=cfg.dl_model.weight_decay,
        force_backend=cfg.dl_model.force_backend,
        use_log_target=False,
        huber_beta=3.0,
        grad_clip=1.0,
        lr_factor=0.5,
        lr_patience=4,
    )

    cv_metrics = None
    cv_error = None
    try:
        cv_metrics = evaluate_groupkfold_cv(
            df=train_df,
            battery_ids=actual_train_batteries,
            n_splits=5,
        )
    except Exception as exc:
        cv_error = f"{type(exc).__name__}: {exc}"
        logger.warning(f"GroupKFold CV failed: {cv_error}")

    # Fit conformal calibrator on held-out calibration batteries.
    # LOBO uses full train rows (actual_train + cal batteries) to refit
    # within-group models for harder groups (cold).
    train_df_for_lobo = df[df["battery_id"].astype(str).isin(train_batteries)].copy()
    calibrator = None
    if cal_df.empty:
        logger.warning("Calibration set is empty; skipping conformal calibration.")
    else:
        calibrator = ConformalCalibrator(coverage=cfg.conformal.coverage, rul_col="RUL")
        calibrator.fit(
            cal_df,
            ml_model,
            train_df=train_df_for_lobo,
            actual_train_batteries=actual_train_batteries,
        )
        calibrator.save(trained_models_dir / "conformal_calibrator.json")

    # Fit non-parametric bootstrap CI on calibration residuals.
    bootstrap_calibrator = None
    adaptive_calibrators = {}   # per temperature group
    if not cal_df.empty:
        try:
            cal_feat = cal_df[FEATURE_COLUMNS].fillna(0)
            cal_pred = ml_model.predict(cal_feat)
            cal_true = pd.to_numeric(cal_df["RUL"], errors="coerce").to_numpy(dtype=float)
            bootstrap_calibrator = BootstrapCalibrator(
                coverage=cfg.conformal.coverage
            ).fit(cal_true, cal_pred)
            bootstrap_calibrator.save(trained_models_dir / "bootstrap_calibrator.json")

            # Per-group adaptive CI: use all non-test batteries in each temp group.
            from src.modeling.conformal import _BATTERY_TEMP_GROUP
            non_test = [b for b in train_batteries if b not in test_batteries]
            non_test_df = df[df["battery_id"].isin(non_test)].copy()
            for grp in ["room", "hot", "cold"]:
                grp_bats = [b for b in non_test if _BATTERY_TEMP_GROUP.get(b) == grp]
                grp_df = non_test_df[non_test_df["battery_id"].isin(grp_bats)]
                if len(grp_df) < 20:
                    continue
                gX = grp_df[FEATURE_COLUMNS].fillna(0)
                gp = ml_model.predict(gX)
                gt = pd.to_numeric(grp_df["RUL"], errors="coerce").to_numpy(dtype=float)
                ac = AdaptiveConformalCalibrator(coverage=cfg.conformal.coverage).fit(gt, gp)
                ac.save(trained_models_dir / f"adaptive_calibrator_{grp}.json")
                adaptive_calibrators[grp] = ac
                logger.info(f"  Adaptive CI fitted for group '{grp}' ({len(grp_bats)} batteries)")
        except Exception as exc:
            logger.warning(f"Bootstrap/adaptive calibrator failed: {exc}")

    # Drift monitoring reference and immediate test-set drift report.
    test_drift_report = None
    drift_error = None
    try:
        drift_monitor = DriftMonitor.fit(train_df)
        drift_monitor.save(trained_models_dir / "feature_distributions.json")
        test_drift_report = drift_monitor.compute(test_df, label="test_set")
    except Exception as exc:
        drift_error = f"{type(exc).__name__}: {exc}"
        logger.warning(f"Drift monitor failed: {drift_error}")

    model_metrics = ml_metrics.to_dict()
    model_metrics["baseline_rmse"] = stat_metrics.rmse
    model_metrics["dl_sequence"] = dl_metrics.to_dict()
    if cv_metrics is not None:
        model_metrics["groupkfold_cv"] = cv_metrics.to_dict()
    elif cv_error is not None:
        model_metrics["groupkfold_cv"] = {
            "strategy": "GroupKFold(groups=battery_id)",
            "error": cv_error,
        }
    uq_weights = _compute_uncertainty_weights(
        stat_rmse=stat_metrics.rmse,
        ml_rmse=ml_metrics.rmse,
        dl_rmse=dl_metrics.rmse,
        exclusion_ratio=1.5,
    )
    logger.info(
        f"  Uncertainty ensemble weights: stat={uq_weights['stat']:.3f}, "
        f"ml={uq_weights['ml']:.3f}, dl={uq_weights['dl']:.3f}"
    )
    if uq_weights["stat"] == 0.0:
        logger.info("  Statistical baseline excluded from uncertainty ensemble (underperforming).")
    if test_drift_report is not None:
        model_metrics["drift_report"] = {
            "overall_status": test_drift_report.overall_status,
            "n_alerts": len(test_drift_report.alerts),
            "alerts": test_drift_report.alerts,
            "per_feature_psi": {
                f.feature: {
                    "psi": round(float(f.psi), 4) if np.isfinite(f.psi) else None,
                    "status": f.status,
                }
                for f in test_drift_report.features
            },
            "psi_thresholds": {
                "amber": cfg.drift.psi_amber_threshold,
                "red": cfg.drift.psi_red_threshold,
            },
        }
    elif drift_error is not None:
        model_metrics["drift_report"] = {
            "overall_status": "ERROR",
            "error": drift_error,
        }

    model_metrics["split_metadata"] = {
        "split_strategy": split_strategy,
        "stratified_by": cfg.split.stratify_by,
        "split_seed": split_seed,
        "train_ratio": cfg.split.train_ratio,
        "train_batteries": train_batteries,
        "actual_train_batteries": actual_train_batteries,
        "cal_batteries": cal_batteries,
        "test_batteries": test_batteries,
        "train_temp_mix": _group_counts(train_batteries),
        "actual_train_temp_mix": _group_counts(actual_train_batteries),
        "cal_temp_mix": _group_counts(cal_batteries),
        "test_temp_mix": _group_counts(test_batteries),
        "pretrain_filter": {
            "enabled": bool(cfg.anomaly.flagged_battery_prefixes),
            "target_battery_prefixes": sorted(cfg.anomaly.flagged_battery_prefixes),
            "capacity_floor_factor": cfg.anomaly.capacity_floor_factor,
            "global_threshold": global_threshold,
            "per_battery_thresholds": thresholds,
            "train_rows_removed": dropped_train_rows,
            "cal_rows_flagged_only": int(cal_df["pretrain_anomaly_flag"].sum()),
            "test_rows_flagged_only": int(test_df["pretrain_anomaly_flag"].sum()),
        },
        "uncertainty_weights": uq_weights,
        "conformal_coverage_target": cfg.conformal.coverage,
    }
    (trained_models_dir / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2),
        encoding="utf-8",
    )
    if test_drift_report is not None:
        drift_report_path = output_dir / "drift_report.json"
        drift_report_path.write_text(
            json.dumps(test_drift_report.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(f"  Drift report -> {drift_report_path}")
    if cv_metrics is not None:
        cv_report_path = output_dir / "groupkfold_cv_report.json"
        cv_report_path.write_text(
            json.dumps(cv_metrics.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(f"  GroupKFold CV report -> {cv_report_path}")
    if test_df.empty:
        logger.critical("Test split is empty after filtering battery IDs.")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("STAGE 4: UNCERTAINTY & RISK")
    logger.info("=" * 50)
    risk_horizon = cfg.risk.horizon_cycles
    full_uncertainty_df = df_model.copy()
    full_uncertainty_df["in_test_set"] = full_uncertainty_df["battery_id"].astype(str).isin(set(test_batteries))
    logger.info(
        f"Running uncertainty on full dataset: rows={len(full_uncertainty_df)} "
        f"batteries={full_uncertainty_df['battery_id'].nunique()} "
        f"(test rows={int(full_uncertainty_df['in_test_set'].sum())})"
    )

    run_uncertainty_analysis(
        df=full_uncertainty_df,
        stat_model=stat_model,
        ml_model=ml_model,
        dl_model=dl_model,
        output_dir=output_dir,
        model_weights=uq_weights,
        horizon=risk_horizon,
        calibrator=calibrator,
    )

    if calibrator is not None:
        conformal_report = calibrator.coverage_report(test_df, ml_model)
        (output_dir / "conformal_coverage_report.json").write_text(
            json.dumps(conformal_report, indent=2),
            encoding="utf-8",
        )

    # Replace conformal CI bounds with per-group adaptive cycle-dependent CI.
    if adaptive_calibrators:
        from src.modeling.conformal import _BATTERY_TEMP_GROUP
        unc_path = output_dir / "uncertainty_estimates.json"
        try:
            unc_data = json.loads(unc_path.read_text(encoding="utf-8"))
            for row in unc_data:
                bat = str(row.get("battery_id", ""))
                grp = _BATTERY_TEMP_GROUP.get(bat, "cold")
                ac  = adaptive_calibrators.get(grp)
                if ac is None:
                    continue
                pred = float(row.get("rul_ensemble_mean") or row.get("rul_median") or 0.0)
                lo, hi = ac.predict_interval(pred)
                row["rul_lower_5"] = round(lo, 2)
                row["rul_upper_95"] = round(hi, 2)
                row["interval_source"] = "adaptive_conformal"
                row["q_hat"] = round(hi - pred, 3)
            unc_path.write_text(json.dumps(unc_data, indent=2), encoding="utf-8")
            logger.info("  Adaptive per-group CI applied (cycle-dependent asymmetric bands).")
        except Exception as exc:
            logger.warning(f"Adaptive CI post-processing failed: {exc}")

    logger.info("=" * 50)
    logger.info("STAGE 4.2: SURVIVAL/HAZARD RISK")
    logger.info("=" * 50)

    # NOTE: Kaplan-Meier is fit and evaluated on test_df only — this is a
    # descriptive analysis of the test population, not a train/test generalization
    # result. KM has no separate training phase, so this is the correct usage.
    survival_preds, survival_metrics = run_survival_risk(
        df=test_df,
        output_dir=output_dir,
        feature_cols=DEFAULT_FEATURES,
        horizon=risk_horizon,
    )
    logger.info(
        f"Saved survival risk outputs for {len(survival_preds)} rows "
        f"across {survival_metrics.n_batteries} batteries"
    )

    logger.info("=" * 50)
    logger.info("STAGE 4.5: ANOMALY DETECTION")
    logger.info("=" * 50)

    anomalies = run_anomaly_detection(
        df=test_df,
        ml_model=ml_model,
        output_dir=output_dir,
    )
    logger.info(f"Detected {len(anomalies)} anomalies")

    logger.info("=" * 50)
    logger.info("STAGE 5: REASONING")
    logger.info("=" * 50)

    run_reasoning_analysis(
        df=test_df,
        ml_model=ml_model,
        feature_columns=FEATURE_COLUMNS,
        output_dir=output_dir,
    )

    logger.info("=" * 50)
    logger.info("STAGE 6: SUPERVISOR REVIEW")
    logger.info("=" * 50)

    reviewer = SupervisorReviewer(
        output_dir,
        trained_models_dir=trained_models_dir,
        rmse_pass_threshold=cfg.supervisor.rmse_pass_threshold,
        cv_rmse_ratio_warn_threshold=cfg.supervisor.cv_rmse_ratio_warn_threshold,
    )
    reviewer.generate_final_report()

    _write_manifest(
        output_dir,
        ml_rmse=ml_metrics.rmse,
        stat_rmse=stat_metrics.rmse,
        n_test_batteries=len(test_batteries),
    )

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

# src/modeling/uncertainty.py
"""
Stage 4: Uncertainty & Risk Quantification

Purpose
-------
- Produce uncertainty-aware RUL predictions
- Assign actionable risk categories (LOW / MEDIUM / HIGH)
- Generate JSON-safe outputs for downstream auditing

Key Guarantees
--------------
- No point predictions without uncertainty
- All numeric outputs are native Python types (JSON-safe)
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers: force JSON-safe native Python types
# ---------------------------------------------------------------------

def _f(x) -> float:
    return float(x)


def _i(x) -> int:
    return int(x)


# Quantile constants for mapping interval width -> sigma under normal assumption.
# Loaded lazily so that a missing config file doesn't break imports.
@functools.lru_cache(maxsize=None)
def _consts() -> SimpleNamespace:
    cfg = get_config()
    q_low = float(cfg.uncertainty.q_low)
    q_high = float(cfg.uncertainty.q_high)
    return SimpleNamespace(
        Z_LOW=float(norm.ppf(q_low)),      # ~ -1.6449
        Z_HIGH=float(norm.ppf(q_high)),    # ~ +1.6449
        DEFAULT_HORIZON=int(cfg.risk.horizon_cycles),
        DEFAULT_MIN_SIGMA=float(cfg.uncertainty.min_sigma),
    )


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class UncertaintyEstimate:
    observation_id: str
    battery_id: str
    cycle_index: int

    rul_statistical: float
    rul_ml: float
    rul_dl: Optional[float]
    rul_ensemble: float

    rul_lower_5: float
    rul_median: float
    rul_upper_95: float

    uncertainty_method: str
    risk_category: str
    failure_probability: float
    interval_source: Optional[str] = None
    q_hat: Optional[float] = None
    in_test_set: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            "observation_id": str(self.observation_id),
            "battery_id": str(self.battery_id),
            "cycle_index": _i(self.cycle_index),
            "rul_statistical": round(_f(self.rul_statistical), 2),
            "rul_ml": round(_f(self.rul_ml), 2),
            "rul_dl": round(_f(self.rul_dl), 2) if self.rul_dl is not None else None,
            "rul_ensemble": round(_f(self.rul_ensemble), 2),
            "rul_ensemble_mean": round(_f(self.rul_ensemble), 2),
            "rul_lower_5": round(_f(self.rul_lower_5), 2),
            "rul_median": round(_f(self.rul_median), 2),
            "rul_upper_95": round(_f(self.rul_upper_95), 2),
            "uncertainty_method": self.uncertainty_method,
            "risk_category": self.risk_category,
            "failure_probability": round(_f(self.failure_probability), 3),
            "interval_source": self.interval_source,
            "q_hat": round(_f(self.q_hat), 3) if self.q_hat is not None else None,
            "in_test_set": bool(self.in_test_set) if self.in_test_set is not None else None,
        }


@dataclass
class RiskSummary:
    total_observations: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int

    mean_rul_prediction: float
    mean_uncertainty_width: float

    coverage_90: float
    calibration_score: float

    def to_dict(self) -> Dict:
        return {
            "total_observations": _i(self.total_observations),
            "risk_distribution": {
                "LOW": _i(self.low_risk_count),
                "MEDIUM": _i(self.medium_risk_count),
                "HIGH": _i(self.high_risk_count),
            },
            "mean_rul_prediction": round(_f(self.mean_rul_prediction), 2),
            "mean_uncertainty_width": round(_f(self.mean_uncertainty_width), 2),
            "coverage_90_percent": round(_f(self.coverage_90) * 100.0, 1),
            "calibration_score": round(_f(self.calibration_score), 3),
        }


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

class UncertaintyQuantifier:
    """
    Combines statistical + ML models to produce uncertainty-aware RUL
    and actionable risk categorization.
    """

    def __init__(
        self,
        stat_model,
        ml_model,
        dl_model=None,
        model_weights: Optional[Dict[str, float]] = None,
        calibrator=None,
    ):
        self.stat_model = stat_model
        self.ml_model = ml_model
        self.dl_model = dl_model
        self.model_weights = self._normalize_weights(model_weights)
        self.calibrator = calibrator

    def _normalize_weights(self, model_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if model_weights is None:
            if self.dl_model is not None:
                return {"stat": 0.20, "ml": 0.45, "dl": 0.35}
            return {"stat": 0.30, "ml": 0.70, "dl": 0.0}

        stat_w = max(0.0, float(model_weights.get("stat", 0.0)))
        ml_w = max(0.0, float(model_weights.get("ml", 0.0)))
        dl_w = max(0.0, float(model_weights.get("dl", 0.0))) if self.dl_model is not None else 0.0
        total = stat_w + ml_w + dl_w

        if total <= 0:
            if self.dl_model is not None:
                return {"stat": 0.20, "ml": 0.45, "dl": 0.35}
            return {"stat": 0.30, "ml": 0.70, "dl": 0.0}

        return {
            "stat": stat_w / total,
            "ml": ml_w / total,
            "dl": dl_w / total,
        }

    def compute_failure_probability(
        self,
        rul_median: float,
        rul_lower_raw: float,
        rul_upper_raw: float,
        horizon: Optional[int] = None,
        min_sigma: Optional[float] = None,
    ) -> float:
        rm = _f(rul_median)
        rl_raw = _f(rul_lower_raw)
        ru = _f(rul_upper_raw)
        c = _consts()
        horizon = int(c.DEFAULT_HORIZON if horizon is None else horizon)
        min_sigma = float(c.DEFAULT_MIN_SIGMA if min_sigma is None else min_sigma)

        width = ru - rl_raw
        if not np.isfinite(width) or width <= 0:
            sigma = max(abs(rm) * 0.3, min_sigma)
        else:
            sigma = max(width / (c.Z_HIGH - c.Z_LOW), min_sigma)

        prob = float(norm.cdf(horizon, loc=rm, scale=sigma))
        return float(np.clip(prob, 1e-6, 1 - 1e-6))

    def categorize_risk(self, failure_probability: float) -> str:
        fp = _f(failure_probability)
        return get_config().risk.risk_category(fp)

    def predict_with_uncertainty(self, df: pd.DataFrame, horizon: Optional[int] = None) -> List[UncertaintyEstimate]:
        horizon = int(_consts().DEFAULT_HORIZON if horizon is None else horizon)
        estimates: List[UncertaintyEstimate] = []
        conformal_df: Optional[pd.DataFrame] = None
        conf_lower_arr = conf_upper_arr = conf_prob_arr = conf_q_arr = conf_group_arr = None
        if self.calibrator is not None:
            try:
                conformal_df = self.calibrator.predict(df, self.ml_model, horizon=horizon)
                conf_lower_arr = pd.to_numeric(conformal_df["rul_lower_conformal"], errors="coerce").to_numpy(dtype=float)
                conf_upper_arr = pd.to_numeric(conformal_df["rul_upper_conformal"], errors="coerce").to_numpy(dtype=float)
                conf_prob_arr = pd.to_numeric(conformal_df["failure_probability"], errors="coerce").to_numpy(dtype=float)
                conf_q_arr = pd.to_numeric(conformal_df["q_hat"], errors="coerce").to_numpy(dtype=float)
                conf_group_arr = conformal_df["temp_group"].astype(str).to_numpy()
            except Exception as exc:
                logger.warning(f"Conformal predictor failed, reverting to quantile intervals: {exc}")
                conformal_df = None

        ml_q: Optional[pd.DataFrame] = None
        if hasattr(self.ml_model, "predict_quantiles"):
            try:
                ml_q = self.ml_model.predict_quantiles(df)
            except Exception:
                ml_q = None

        if ml_q is not None and {"q05", "q50", "q95"}.issubset(set(ml_q.columns)):
            ml_lower_raw_arr = pd.to_numeric(ml_q["q05"], errors="coerce").to_numpy(dtype=float)
            ml_median_arr = pd.to_numeric(ml_q["q50"], errors="coerce").to_numpy(dtype=float)
            ml_upper_raw_arr = pd.to_numeric(ml_q["q95"], errors="coerce").to_numpy(dtype=float)
        else:
            ml_lower_clip, ml_median_arr, ml_upper_clip = self.ml_model.predict_with_uncertainty(df)
            ml_lower_raw_arr = np.asarray(ml_lower_clip, dtype=float)
            ml_median_arr = np.asarray(ml_median_arr, dtype=float)
            ml_upper_raw_arr = np.asarray(ml_upper_clip, dtype=float)

        for arr in [ml_lower_raw_arr, ml_median_arr, ml_upper_raw_arr]:
            bad = ~np.isfinite(arr)
            if bad.any():
                fill = float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
                arr[bad] = fill

        dl_bounds_raw = None
        if self.dl_model is not None:
            if hasattr(self.dl_model, "predict_with_uncertainty_raw"):
                try:
                    dl_bounds_raw = self.dl_model.predict_with_uncertainty_raw(df)
                except Exception:
                    dl_bounds_raw = None
            if dl_bounds_raw is None:
                try:
                    dl_bounds_raw = self.dl_model.predict_with_uncertainty(df)
                except Exception:
                    dl_bounds_raw = None

        if dl_bounds_raw is not None:
            dl_lower_raw_arr, dl_median_arr, dl_upper_raw_arr = dl_bounds_raw
            dl_lower_raw_arr = np.asarray(dl_lower_raw_arr, dtype=float)
            dl_median_arr = np.asarray(dl_median_arr, dtype=float)
            dl_upper_raw_arr = np.asarray(dl_upper_raw_arr, dtype=float)
        else:
            dl_lower_raw_arr = dl_median_arr = dl_upper_raw_arr = None

        stat_lower_arr, stat_median_arr, stat_upper_arr = self.stat_model.predict_rul_batch_with_uncertainty(
            df["battery_id"].astype(str).to_numpy(),
            pd.to_numeric(df["cycle_index"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["capacity"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["eol_capacity_threshold"], errors="coerce").to_numpy(dtype=float),
        )

        for pos, (_, row) in enumerate(df.iterrows()):
            battery_id = str(row["battery_id"])
            cycle_index = _i(row["cycle_index"])

            stat_lo_clip = stat_lower_arr[pos]
            stat_med = stat_median_arr[pos]
            stat_hi_clip = stat_upper_arr[pos]
            stat_lo_raw = _f(stat_lo_clip)
            stat_hi_raw = _f(stat_hi_clip)
            stat_lo = max(stat_lo_raw, 0.0)
            stat_hi = max(stat_hi_raw, _f(stat_med))

            ml_lo_raw = _f(ml_lower_raw_arr[pos])
            ml_med = _f(ml_median_arr[pos])
            ml_hi_raw = _f(ml_upper_raw_arr[pos])
            if ml_hi_raw < ml_lo_raw:
                ml_lo_raw, ml_hi_raw = ml_hi_raw, ml_lo_raw
            ml_lo = max(ml_lo_raw, 0.0)
            ml_hi = max(ml_hi_raw, ml_med)

            dl_med: Optional[float] = None
            if dl_lower_raw_arr is not None and dl_median_arr is not None and dl_upper_raw_arr is not None:
                dl_lo_raw = _f(dl_lower_raw_arr[pos])
                dl_med = _f(dl_median_arr[pos])
                dl_hi_raw = _f(dl_upper_raw_arr[pos])
                if dl_hi_raw < dl_lo_raw:
                    dl_lo_raw, dl_hi_raw = dl_hi_raw, dl_lo_raw
                dl_lo = max(dl_lo_raw, 0.0)
                dl_hi = max(dl_hi_raw, dl_med)
            else:
                dl_lo_raw = None
                dl_hi_raw = None
                dl_lo = None
                dl_hi = None

            components = [
                ("stat", _f(stat_med), stat_lo, stat_hi, stat_lo_raw, stat_hi_raw),
                ("ml", ml_med, ml_lo, ml_hi, ml_lo_raw, ml_hi_raw),
            ]
            if (
                dl_med is not None
                and dl_lo is not None
                and dl_hi is not None
                and dl_lo_raw is not None
                and dl_hi_raw is not None
            ):
                components.append(("dl", dl_med, dl_lo, dl_hi, dl_lo_raw, dl_hi_raw))

            active = [
                (name, med, lo, hi, lo_raw, hi_raw)
                for name, med, lo, hi, lo_raw, hi_raw in components
                if self.model_weights.get(name, 0.0) > 0 and np.isfinite(med)
            ]
            if not active:
                # Fallback safety: ensure at least ML contributes.
                active = [("ml", ml_med, ml_lo, ml_hi, ml_lo_raw, ml_hi_raw)]

            den = float(sum(self.model_weights.get(name, 0.0) for name, *_ in active))
            if den <= 0:
                rul_ensemble = float(np.mean([med for _, med, *_ in active]))
                rul_med = rul_ensemble
            else:
                rul_ensemble = float(
                    sum(self.model_weights.get(name, 0.0) * med for name, med, *_ in active) / den
                )
                rul_med = rul_ensemble

            rul_lower_raw = min(lo_raw for _, _, _, _, lo_raw, _ in active)
            rul_upper_raw = max(hi_raw for _, _, _, _, _, hi_raw in active)
            rul_lower = max(0.0, rul_lower_raw)
            rul_upper = max(rul_med, rul_upper_raw)

            w = self.model_weights
            if conformal_df is not None and conf_lower_arr is not None and conf_upper_arr is not None:
                c_lo = conf_lower_arr[pos] if pos < len(conf_lower_arr) else np.nan
                c_hi = conf_upper_arr[pos] if pos < len(conf_upper_arr) else np.nan
                c_p = conf_prob_arr[pos] if conf_prob_arr is not None and pos < len(conf_prob_arr) else np.nan
                c_q = conf_q_arr[pos] if conf_q_arr is not None and pos < len(conf_q_arr) else np.nan
                c_g = conf_group_arr[pos] if conf_group_arr is not None and pos < len(conf_group_arr) else "unknown"

                if np.isfinite(c_lo) and np.isfinite(c_hi):
                    rul_lower = max(0.0, float(c_lo))
                    rul_upper = max(rul_lower, float(c_hi))
                    if np.isfinite(c_p):
                        fail_prob = float(np.clip(c_p, 1e-6, 1 - 1e-6))
                    else:
                        # Fallback if conformal probability is missing.
                        raw_lo = rul_med - (float(c_q) if np.isfinite(c_q) else 0.0)
                        raw_hi = rul_med + (float(c_q) if np.isfinite(c_q) else 0.0)
                        fail_prob = self.compute_failure_probability(
                            rul_median=rul_med,
                            rul_lower_raw=raw_lo,
                            rul_upper_raw=raw_hi,
                            horizon=horizon,
                        )
                    risk = self.categorize_risk(fail_prob)
                    uncertainty_method = (
                        "conformal_weighted_ensemble"
                        f"(group={c_g},q_hat={float(c_q) if np.isfinite(c_q) else float('nan'):.2f},"
                        f"stat={w.get('stat',0.0):.2f},ml={w.get('ml',0.0):.2f},dl={w.get('dl',0.0):.2f})"
                    )
                    q_hat_out = float(c_q) if np.isfinite(c_q) else None
                    interval_source = "conformal"
                else:
                    # Conformal rows missing -> fallback path.
                    fail_prob = self.compute_failure_probability(
                        rul_median=rul_med,
                        rul_lower_raw=rul_lower_raw,
                        rul_upper_raw=rul_upper_raw,
                        horizon=horizon,
                    )
                    risk = self.categorize_risk(fail_prob)
                    uncertainty_method = (
                        "quantile_regression_bootstrap_weighted_ensemble"
                        f"(stat={w.get('stat',0.0):.2f},ml={w.get('ml',0.0):.2f},dl={w.get('dl',0.0):.2f})"
                    )
                    q_hat_out = None
                    interval_source = "quantile_regression"
            else:
                fail_prob = self.compute_failure_probability(
                    rul_median=rul_med,
                    rul_lower_raw=rul_lower_raw,
                    rul_upper_raw=rul_upper_raw,
                    horizon=horizon,
                )
                risk = self.categorize_risk(fail_prob)
                uncertainty_method = (
                    "quantile_regression_bootstrap_weighted_ensemble"
                    f"(stat={w.get('stat',0.0):.2f},ml={w.get('ml',0.0):.2f},dl={w.get('dl',0.0):.2f})"
                )
                q_hat_out = None
                interval_source = "quantile_regression"

            estimates.append(
                UncertaintyEstimate(
                    observation_id=f"{battery_id}_cycle_{cycle_index}",
                    battery_id=battery_id,
                    cycle_index=cycle_index,
                    rul_statistical=_f(stat_med),
                    rul_ml=ml_med,
                    rul_dl=dl_med,
                    rul_ensemble=_f(rul_ensemble),
                    rul_lower_5=_f(rul_lower),
                    rul_median=_f(rul_med),
                    rul_upper_95=_f(rul_upper),
                    uncertainty_method=uncertainty_method,
                    risk_category=risk,
                    failure_probability=_f(fail_prob),
                    interval_source=interval_source,
                    q_hat=q_hat_out,
                    in_test_set=(
                        bool(row["in_test_set"])
                        if "in_test_set" in row and pd.notna(row["in_test_set"])
                        else None
                    ),
                )
            )

        return estimates

    def compute_risk_summary(
        self,
        estimates: List[UncertaintyEstimate],
        actuals: Optional[np.ndarray] = None,
    ) -> RiskSummary:
        n = len(estimates)

        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        preds, widths = [], []
        covered = 0

        for i, e in enumerate(estimates):
            counts[e.risk_category] += 1
            preds.append(_f(e.rul_ensemble))
            widths.append(_f(e.rul_upper_95) - _f(e.rul_lower_5))

            if actuals is not None:
                a = _f(actuals[i])
                if _f(e.rul_lower_5) <= a <= _f(e.rul_upper_95):
                    covered += 1

        coverage = covered / n if actuals is not None and n else 0.0
        target_coverage = float(get_config().conformal.coverage)
        calibration = 1.0 - abs(coverage - target_coverage) if actuals is not None else 0.0

        return RiskSummary(
            total_observations=_i(n),
            low_risk_count=_i(counts["LOW"]),
            medium_risk_count=_i(counts["MEDIUM"]),
            high_risk_count=_i(counts["HIGH"]),
            mean_rul_prediction=_f(np.mean(preds)) if preds else float("nan"),
            mean_uncertainty_width=_f(np.mean(widths)) if widths else float("nan"),
            coverage_90=_f(coverage),
            calibration_score=_f(calibration),
        )


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_uncertainty_analysis(
    df: pd.DataFrame,
    stat_model,
    ml_model,
    dl_model,
    output_dir: Path,
    model_weights: Optional[Dict[str, float]] = None,
    horizon: Optional[int] = None,
    calibrator=None,
) -> Tuple[List[UncertaintyEstimate], RiskSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(_consts().DEFAULT_HORIZON if horizon is None else horizon)

    logger.info("Running uncertainty quantification...")

    uq = UncertaintyQuantifier(
        stat_model,
        ml_model,
        dl_model=dl_model,
        model_weights=model_weights,
        calibrator=calibrator,
    )
    estimates = uq.predict_with_uncertainty(df, horizon=horizon)
    risk_summary = uq.compute_risk_summary(estimates, df["RUL"].values)

    (output_dir / "uncertainty_estimates.json").write_text(
        json.dumps([e.to_dict() for e in estimates], indent=2),
        encoding="utf-8",
    )

    (output_dir / "uncertainty_metrics.json").write_text(
        json.dumps(risk_summary.to_dict(), indent=2),
        encoding="utf-8",
    )

    return estimates, risk_summary

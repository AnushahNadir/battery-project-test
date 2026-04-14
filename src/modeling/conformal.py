"""
Stratified split conformal prediction for RUL regression.

v2 highlights
-------------
- Uses LOBO (leave-one-battery-out) calibration for heterogeneous groups
  (default: cold) to reduce under-coverage when fixed calibration batteries
  are not representative.
- Keeps split conformal for stable groups (room/hot).
- Applies safety inflation when a group has too few calibration batteries.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


_BATTERY_TEMP_GROUP: dict[str, str] = {
    **{f"B00{n:02d}": "room" for n in [5, 6, 7, 18]},
    **{f"B00{n:02d}": "room" for n in range(25, 29)},
    **{f"B00{n:02d}": "hot" for n in range(29, 33)},
    **{f"B00{n:02d}": "room" for n in [33, 34, 36]},
    **{f"B00{n:02d}": "hot" for n in range(38, 41)},
    **{f"B00{n:02d}": "cold" for n in range(41, 57)},
}

TEMP_GROUPS = ("room", "hot", "cold")


def _get_temp_group(battery_id: str) -> str:
    return _BATTERY_TEMP_GROUP.get(str(battery_id), "room")


def select_calibration_batteries(
    train_batteries: list[str],
    n_cal: int = 6,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Carve n_cal batteries from training batteries for conformal calibration.

    Returns:
        calibration_batteries, actual_train_batteries
    """
    rng = random.Random(random_seed)
    train_ids = sorted(set(str(b) for b in train_batteries))
    if len(train_ids) < 3:
        raise ValueError("Need at least 3 training batteries for calibration split.")

    groups: dict[str, list[str]] = defaultdict(list)
    for bid in train_ids:
        groups[_get_temp_group(bid)].append(bid)

    non_empty = [g for g in TEMP_GROUPS if groups[g]]
    n_cal = max(len(non_empty), min(int(n_cal), len(train_ids) - 1))

    alloc: dict[str, int] = {}
    total = len(train_ids)
    remaining = n_cal
    for i, grp in enumerate(non_empty):
        n_grp = len(groups[grp])
        max_take = max(1, n_grp - 1)
        if i == len(non_empty) - 1:
            take = max(1, min(remaining, max_take))
        else:
            raw = int(round(n_cal * n_grp / total))
            min_left = len(non_empty) - i - 1
            take = max(1, raw)
            take = min(take, max_take)
            take = min(take, max(1, remaining - min_left))
        alloc[grp] = take
        remaining -= take

    cal_ids: list[str] = []
    for grp in TEMP_GROUPS:
        if not groups[grp]:
            continue
        cands = sorted(groups[grp])
        rng.shuffle(cands)
        cal_ids.extend(cands[: alloc.get(grp, 1)])

    cal_set = set(cal_ids)
    actual_train = [b for b in train_ids if b not in cal_set]
    if not actual_train:
        moved = sorted(cal_set)[-1]
        cal_set.remove(moved)
        actual_train.append(moved)

    cal_out = sorted(cal_set)
    train_out = sorted(actual_train)
    cfg = get_config()
    lobo_groups = frozenset(cfg.conformal.lobo_groups)

    logger.info("\n[Conformal calibration split]")
    for grp in TEMP_GROUPS:
        strategy = "LOBO" if grp in lobo_groups else "split"
        grp_cal = [b for b in cal_out if _get_temp_group(b) == grp]
        grp_train = [b for b in train_out if _get_temp_group(b) == grp]
        if grp_cal or grp_train:
            logger.info(f"  {grp:>5} [{strategy}]: cal={grp_cal}  actual_train={grp_train}")

    return cal_out, train_out


def _nonconformity_scores(df: pd.DataFrame, model, rul_col: str = "RUL") -> np.ndarray:
    """
    Absolute residual nonconformity score |y - y_hat|.
    """
    feature_cols = [c for c in model.feature_columns if c in df.columns]
    y_pred = np.asarray(model.predict(df[feature_cols]), dtype=float)
    col = rul_col if rul_col in df.columns else "rul"
    y_true = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    out = np.full(len(df), np.nan, dtype=float)
    out[good] = np.abs(y_true[good] - y_pred[good])
    return out


def _clone_model(model):
    """
    Clone model instance for LOBO refits.
    """
    try:
        return deepcopy(model)
    except Exception:
        # Fallback for non-deepcopyable models.
        cls = type(model)
        if hasattr(model, "feature_columns"):
            return cls(feature_columns=list(model.feature_columns))
        return cls()


def _lobo_scores(
    group_df: pd.DataFrame,
    model,
    train_df: pd.DataFrame,
    actual_train_batteries: list[str],
    rul_col: str = "RUL",
) -> np.ndarray:
    """
    LOBO calibration scores for one group.

    For each calibration battery in this group:
      - fit on actual_train + (other calibration batteries in group)
      - score on the left-out battery
    """
    group_batteries = sorted(group_df["battery_id"].astype(str).unique())
    all_scores: list[float] = []

    logger.info(
        f"    [LOBO] calibrating on {len(group_batteries)} batteries "
        f"(leave-one-out): {group_batteries}"
    )

    for leave_out in group_batteries:
        # Ensure leave_out is excluded even if it exists in actual_train_batteries.
        base_train = [b for b in actual_train_batteries if b != leave_out]
        extra = [b for b in group_batteries if b != leave_out and b not in base_train]
        lobo_train_bats = base_train + extra
        lobo_train_df = train_df[train_df["battery_id"].astype(str).isin(lobo_train_bats)].copy()
        lobo_test_df = group_df[group_df["battery_id"].astype(str) == leave_out].copy()
        if lobo_train_df.empty or lobo_test_df.empty:
            continue

        lobo_model = _clone_model(model)
        try:
            lobo_model.fit(lobo_train_df, quantiles=True)
        except TypeError:
            lobo_model.fit(lobo_train_df)
        except Exception as exc:
            warnings.warn(f"LOBO refit failed for leave_out={leave_out}: {exc}", UserWarning, stacklevel=3)
            continue

        scores = _nonconformity_scores(lobo_test_df, lobo_model, rul_col=rul_col)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            continue
        all_scores.extend(scores.tolist())
        logger.info(
            f"      leave_out={leave_out}: {len(scores)} scores, "
            f"mean_residual={float(np.mean(scores)):.1f}, max={float(np.max(scores)):.1f}"
        )

    return np.asarray(all_scores, dtype=float)


@dataclass
class GroupCalibration:
    group: str
    n_cal: int
    n_cal_batteries: int
    scores: list[float]
    q_hat: float
    coverage: float
    strategy: str
    inflated: bool = False


@dataclass
class ConformalCalibrator:
    """
    Stratified conformal calibrator.
    """

    coverage: float = field(default_factory=lambda: float(get_config().conformal.coverage))
    rul_col: str = "RUL"
    _group_cal: dict[str, GroupCalibration] = field(default_factory=dict, init=False, repr=False)
    _global_q_hat: float = field(default=np.inf, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        cal_df: pd.DataFrame,
        model,
        train_df: Optional[pd.DataFrame] = None,
        actual_train_batteries: Optional[list[str]] = None,
    ) -> "ConformalCalibrator":
        if cal_df.empty:
            raise ValueError("Calibration DataFrame is empty.")

        rul_col = self.rul_col if self.rul_col in cal_df.columns else "rul"
        alpha = 1.0 - float(self.coverage)
        cfg = get_config()
        lobo_groups = frozenset(cfg.conformal.lobo_groups)
        min_cal_batteries = int(cfg.conformal.min_cal_batteries)
        safety_factor = float(cfg.conformal.safety_factor)
        work = cal_df.copy()
        work["_group"] = work["battery_id"].astype(str).map(_get_temp_group)

        all_scores: list[float] = []
        logger.info(f"\n[Conformal calibration  coverage={self.coverage:.0%}]")

        for grp in TEMP_GROUPS:
            grp_df = work[work["_group"] == grp].copy()
            if grp_df.empty:
                continue
            grp_batteries = sorted(grp_df["battery_id"].astype(str).unique())
            strategy = "lobo" if grp in lobo_groups else "split"

            if strategy == "lobo":
                if train_df is None or actual_train_batteries is None:
                    warnings.warn(
                        f"LOBO requested for group '{grp}' but train_df/actual_train_batteries missing. "
                        "Falling back to split.",
                        UserWarning,
                        stacklevel=2,
                    )
                    strategy = "split"
                    scores = _nonconformity_scores(grp_df, model, rul_col=rul_col)
                else:
                    # Use all training batteries in this group (not only fixed cal batteries)
                    # to capture within-group heterogeneity.
                    train_group_df = train_df[
                        train_df["battery_id"].astype(str).map(_get_temp_group) == grp
                    ].copy()
                    if train_group_df.empty:
                        train_group_df = grp_df.copy()
                    grp_batteries = sorted(train_group_df["battery_id"].astype(str).unique())
                    scores = _lobo_scores(
                        train_group_df,
                        model,
                        train_df=train_df,
                        actual_train_batteries=actual_train_batteries,
                        rul_col=rul_col,
                    )
            else:
                scores = _nonconformity_scores(grp_df, model, rul_col=rul_col)

            scores = scores[np.isfinite(scores)]
            n = int(len(scores))
            if n == 0:
                warnings.warn(f"No valid scores for group '{grp}'.", UserWarning, stacklevel=2)
                continue

            all_scores.extend(scores.tolist())
            level = min(float(np.ceil((n + 1) * (1 - alpha)) / n), 1.0)
            q_hat = float(np.quantile(scores, level))

            inflated = False
            if len(grp_batteries) < min_cal_batteries:
                q_hat *= safety_factor
                inflated = True
                warnings.warn(
                    f"Group '{grp}' has only {len(grp_batteries)} calibration battery. "
                    f"Applied {safety_factor:.0%} safety inflation to q_hat.",
                    UserWarning,
                    stacklevel=2,
                )

            self._group_cal[grp] = GroupCalibration(
                group=grp,
                n_cal=n,
                n_cal_batteries=len(grp_batteries),
                scores=scores.tolist(),
                q_hat=q_hat,
                coverage=float(self.coverage),
                strategy=strategy,
                inflated=inflated,
            )

            logger.info(
                f"  {grp:>5} [{strategy}]  n_scores={n:4d}  n_batteries={len(grp_batteries)}  "
                f"q_hat={q_hat:.2f}{'*' if inflated else ''}  "
                f"score_p50={float(np.median(scores)):.1f}  score_p90={float(np.quantile(scores, 0.9)):.1f}"
            )

        if not all_scores:
            raise ValueError("No valid calibration scores could be computed.")

        n_all = len(all_scores)
        level_all = min(float(np.ceil((n_all + 1) * (1 - alpha)) / n_all), 1.0)
        self._global_q_hat = float(np.quantile(np.asarray(all_scores, dtype=float), level_all))
        self._fitted = True
        logger.info(f"  global fallback q_hat = {self._global_q_hat:.2f}")
        return self

    def q_hat_for(self, battery_id: str) -> float:
        if not self._fitted:
            raise RuntimeError("ConformalCalibrator not fitted.")
        grp = _get_temp_group(str(battery_id))
        if grp in self._group_cal:
            return float(self._group_cal[grp].q_hat)
        return float(self._global_q_hat)

    def predict(self, df: pd.DataFrame, model, horizon: Optional[int] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ConformalCalibrator not fitted.")

        cfg = get_config()
        risk_cfg = cfg.risk
        horizon = int(risk_cfg.horizon_cycles if horizon is None else horizon)
        feature_cols = [c for c in model.feature_columns if c in df.columns]
        preds = np.asarray(model.predict(df[feature_cols]), dtype=float)
        rows = []

        for i, (_, row) in enumerate(df.iterrows()):
            bid = str(row.get("battery_id", "unknown"))
            cyc = int(row.get("cycle_index", i))
            pred = float(preds[i]) if i < len(preds) and np.isfinite(preds[i]) else 0.0
            q_hat = self.q_hat_for(bid)
            grp = _get_temp_group(bid)

            lo_raw = pred - q_hat
            hi_raw = pred + q_hat
            lo = max(0.0, lo_raw)
            hi = max(0.0, hi_raw)
            width = hi - lo

            # Uniform mass over [lo_raw, hi_raw].
            if hi_raw <= lo_raw:
                fail_prob = 1.0 if pred <= horizon else 0.0
            else:
                fail_prob = float(np.clip((min(horizon, hi_raw) - lo_raw) / (hi_raw - lo_raw), 0.0, 1.0))

            risk = risk_cfg.risk_category(float(fail_prob))

            rows.append(
                {
                    "battery_id": bid,
                    "cycle_index": cyc,
                    "rul_pred": pred,
                    "rul_lower_conformal": lo,
                    "rul_upper_conformal": hi,
                    "q_hat": q_hat,
                    "temp_group": grp,
                    "interval_width": width,
                    "failure_probability": fail_prob,
                    "risk_category": risk,
                    "horizon_cycles": int(horizon),
                }
            )
        return pd.DataFrame(rows)

    def coverage_report(self, test_df: pd.DataFrame, model) -> dict:
        if not self._fitted:
            raise RuntimeError("ConformalCalibrator not fitted.")

        rul_col = self.rul_col if self.rul_col in test_df.columns else "rul"
        intervals = self.predict(test_df, model)
        y_true = pd.to_numeric(test_df[rul_col], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(intervals["rul_lower_conformal"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(intervals["rul_upper_conformal"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
        covered = valid & (y_true >= lo) & (y_true <= hi)
        overall = float(covered[valid].mean()) if valid.any() else float("nan")

        report: dict = {
            "target_coverage": float(self.coverage),
            "overall_empirical_coverage": round(overall, 4) if np.isfinite(overall) else None,
            "n_test_rows": int(len(test_df)),
            "per_group": {},
        }

        work = test_df.copy()
        work["_valid"] = valid
        work["_covered"] = covered
        work["_group"] = work["battery_id"].astype(str).map(_get_temp_group)

        logger.info(f"\n[Conformal coverage report  target={self.coverage:.0%}]")
        if report["overall_empirical_coverage"] is not None:
            logger.info(f"  Overall: {report['overall_empirical_coverage']:.1%}")
        for grp in TEMP_GROUPS:
            gdf = work[work["_group"] == grp]
            if gdf.empty:
                continue
            gvalid = gdf["_valid"].to_numpy(dtype=bool)
            emp = float(gdf.loc[gvalid, "_covered"].mean()) if gvalid.any() else float("nan")
            gc = self._group_cal.get(grp)
            q_hat = float(gc.q_hat) if gc is not None else float(self._global_q_hat)
            strategy = gc.strategy if gc is not None else "fallback"
            gap = emp - float(self.coverage) if np.isfinite(emp) else float("nan")
            report["per_group"][grp] = {
                "n_rows": int(len(gdf)),
                "empirical_coverage": round(emp, 4) if np.isfinite(emp) else None,
                "q_hat": round(q_hat, 2),
                "strategy": strategy,
                "gap_vs_target": round(gap, 4) if np.isfinite(gap) else None,
            }
            if np.isfinite(emp):
                flag = "OK" if emp >= self.coverage else "UNDER-COVERED"
                logger.info(
                    f"  {grp:>5} [{strategy}]: {emp:.1%}  "
                    f"(gap={gap:+.1%}  q_hat={q_hat:.1f}  n={len(gdf)})  {flag}"
                )
            else:
                logger.info(f"  {grp:>5} [{strategy}]: empirical=NA  n={len(gdf)}")
        return report

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "coverage": float(self.coverage),
            "rul_col": self.rul_col,
            "global_q_hat": float(self._global_q_hat),
            "group_cal": {
                grp: {
                    "group": g.group,
                    "n_cal": int(g.n_cal),
                    "n_cal_batteries": int(g.n_cal_batteries),
                    "scores": [float(s) for s in g.scores],
                    "q_hat": float(g.q_hat),
                    "coverage": float(g.coverage),
                    "strategy": g.strategy,
                    "inflated": bool(g.inflated),
                }
                for grp, g in self._group_cal.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"  Saved conformal calibrator -> {path}")

    @classmethod
    def load(cls, path: Path) -> "ConformalCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(coverage=float(data["coverage"]), rul_col=str(data["rul_col"]))
        obj._global_q_hat = float(data["global_q_hat"])
        for grp, g in data.get("group_cal", {}).items():
            obj._group_cal[grp] = GroupCalibration(
                group=str(g["group"]),
                n_cal=int(g["n_cal"]),
                n_cal_batteries=int(g["n_cal_batteries"]),
                scores=[float(s) for s in g["scores"]],
                q_hat=float(g["q_hat"]),
                coverage=float(g["coverage"]),
                strategy=str(g["strategy"]),
                inflated=bool(g.get("inflated", False)),
            )
        obj._fitted = True
        return obj

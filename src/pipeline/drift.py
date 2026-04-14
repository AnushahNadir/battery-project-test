"""
Feature drift monitoring using Population Stability Index (PSI).

This module compares incoming data against a training reference distribution.
PSI thresholds:
  - PSI < 0.10: GREEN
  - 0.10 <= PSI < 0.20: AMBER
  - PSI >= 0.20: RED

Design note:
Condition features can indicate protocol/data drift and should alert.
Degradation features are expected to shift with battery age and are logged
for diagnostics but do not trigger alerts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


_CFG = get_config()
PSI_AMBER = float(_CFG.drift.psi_amber_threshold)
PSI_RED = float(_CFG.drift.psi_red_threshold)
N_BINS = int(_CFG.drift.n_bins)

CONDITION_FEATURES = [
    "temp_mean",
    "temp_max",
    "v_min",
    "v_mean",
    "i_mean",
    "i_min",
    "duration_s",
]

DEGRADATION_FEATURES = [
    "capacity",
    "ah_est",
    "energy_j",
]

DRIFT_FEATURES = CONDITION_FEATURES + DEGRADATION_FEATURES


def _psi_status(psi: float) -> str:
    if not np.isfinite(psi):
        return "UNKNOWN"
    if psi >= PSI_RED:
        return "RED"
    if psi >= PSI_AMBER:
        return "AMBER"
    return "GREEN"


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bin_edges: np.ndarray) -> float:
    """
    Compute PSI using quantile edges fit on expected distribution.

    Parameters
    ----------
    expected:
        Reference/training values.
    actual:
        Incoming values to compare against expected.
    bin_edges:
        Quantile edges including the 0th and 100th percentile; outer
        edges are extended to [-inf, +inf] so out-of-range values are
        retained in edge bins.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    interior = bin_edges[1:-1]
    full_edges = np.concatenate([[-np.inf], interior, [np.inf]])

    e_counts, _ = np.histogram(expected, bins=full_edges)
    a_counts, _ = np.histogram(actual, bins=full_edges)

    e_pct = np.clip(e_counts / max(len(expected), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(len(actual), 1), 1e-6, None)
    e_pct = e_pct / e_pct.sum()
    a_pct = a_pct / a_pct.sum()

    m = min(len(e_pct), len(a_pct))
    e_pct = e_pct[:m]
    a_pct = a_pct[:m]

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return psi


@dataclass
class FeatureDistribution:
    feature: str
    bin_edges: list[float]
    expected_pcts: list[float]
    n_train: int
    mean: float
    std: float
    p5: float
    p50: float
    p95: float

    @classmethod
    def fit(cls, feature: str, values: np.ndarray) -> "FeatureDistribution":
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"No finite training values for '{feature}'")

        edges = np.quantile(values, np.linspace(0.0, 1.0, N_BINS + 1))
        edges = np.unique(edges)

        interior = edges[1:-1]
        full_edges = np.concatenate([[-np.inf], interior, [np.inf]])
        counts, _ = np.histogram(values, bins=full_edges)
        pcts = counts / max(counts.sum(), 1)

        return cls(
            feature=feature,
            bin_edges=edges.astype(float).tolist(),
            expected_pcts=pcts.astype(float).tolist(),
            n_train=int(len(values)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            p5=float(np.quantile(values, 0.05)),
            p50=float(np.quantile(values, 0.50)),
            p95=float(np.quantile(values, 0.95)),
        )

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "bin_edges": self.bin_edges,
            "expected_pcts": self.expected_pcts,
            "n_train": self.n_train,
            "mean": self.mean,
            "std": self.std,
            "p5": self.p5,
            "p50": self.p50,
            "p95": self.p95,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureDistribution":
        return cls(
            feature=str(payload["feature"]),
            bin_edges=[float(x) for x in payload["bin_edges"]],
            expected_pcts=[float(x) for x in payload["expected_pcts"]],
            n_train=int(payload["n_train"]),
            mean=float(payload["mean"]),
            std=float(payload["std"]),
            p5=float(payload["p5"]),
            p50=float(payload["p50"]),
            p95=float(payload["p95"]),
        )


@dataclass
class FeatureDriftResult:
    feature: str
    psi: float
    status: str
    n_actual: int
    train_mean: float
    actual_mean: float
    mean_shift: float
    alert_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4) if np.isfinite(self.psi) else None,
            "status": self.status,
            "n_actual": self.n_actual,
            "train_mean": round(self.train_mean, 4),
            "actual_mean": round(self.actual_mean, 4),
            "mean_shift": round(self.mean_shift, 4),
            "alert": self.alert_message,
        }


@dataclass
class DriftReport:
    overall_status: str
    n_rows: int
    n_batteries: int
    features: list[FeatureDriftResult]
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "n_rows": self.n_rows,
            "n_batteries": self.n_batteries,
            "psi_thresholds": {"amber": PSI_AMBER, "red": PSI_RED},
            "alerts": self.alerts,
            "features": [x.to_dict() for x in self.features],
        }

    def log(self, label: str = "") -> None:
        prefix = f"[DriftMonitor {label}]".strip()
        logger.info(f"\n{prefix} overall={self.overall_status} rows={self.n_rows} batteries={self.n_batteries}")
        for item in sorted(self.features, key=lambda r: (0 if r.status == "RED" else 1, -(r.psi if np.isfinite(r.psi) else -1))):
            psi_txt = f"{item.psi:.4f}" if np.isfinite(item.psi) else "nan"
            logger.info(
                f"  {item.feature:<12} psi={psi_txt:>7} status={item.status:<7} "
                f"train_mu={item.train_mean:>7.3f} actual_mu={item.actual_mean:>7.3f} "
                f"shift={item.mean_shift:+7.3f}"
            )
        if self.alerts:
            logger.info(f"  alerts={len(self.alerts)}")
            for msg in self.alerts:
                logger.info(f"    - {msg}")


class DriftMonitor:
    def __init__(self) -> None:
        self._distributions: dict[str, FeatureDistribution] = {}
        self._fitted = False

    @classmethod
    def fit(cls, train_df: pd.DataFrame, features: list[str] | None = None) -> "DriftMonitor":
        monitor = cls()
        selected = features or [f for f in DRIFT_FEATURES if f in train_df.columns]
        for feat in selected:
            vals = pd.to_numeric(train_df[feat], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            monitor._distributions[feat] = FeatureDistribution.fit(feat, vals)
        monitor._fitted = True
        return monitor

    def save(self, path: Path) -> None:
        payload = {
            "psi_thresholds": {"amber": PSI_AMBER, "red": PSI_RED},
            "n_bins": N_BINS,
            "features": {k: v.to_dict() for k, v in self._distributions.items()},
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"  [DriftMonitor] saved reference distributions -> {path}")

    @classmethod
    def load(cls, path: Path) -> "DriftMonitor":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        monitor = cls()
        for feat, payload in data.get("features", {}).items():
            monitor._distributions[feat] = FeatureDistribution.from_dict(payload)
        monitor._fitted = True
        return monitor

    def compute(self, new_df: pd.DataFrame, label: str = "") -> DriftReport:
        if not self._fitted:
            raise RuntimeError("DriftMonitor must be fitted before compute().")

        feature_results: list[FeatureDriftResult] = []
        alerts: list[str] = []

        for feat, dist in self._distributions.items():
            if feat not in new_df.columns:
                feature_results.append(
                    FeatureDriftResult(
                        feature=feat,
                        psi=float("nan"),
                        status="UNKNOWN",
                        n_actual=0,
                        train_mean=dist.mean,
                        actual_mean=float("nan"),
                        mean_shift=float("nan"),
                        alert_message=f"Feature '{feat}' missing in incoming data",
                    )
                )
                continue

            vals = pd.to_numeric(new_df[feat], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                feature_results.append(
                    FeatureDriftResult(
                        feature=feat,
                        psi=float("nan"),
                        status="UNKNOWN",
                        n_actual=0,
                        train_mean=dist.mean,
                        actual_mean=float("nan"),
                        mean_shift=float("nan"),
                        alert_message=f"Feature '{feat}' has no valid incoming values",
                    )
                )
                continue

            edges = np.array(dist.bin_edges, dtype=float)
            interior = edges[1:-1]
            full_edges = np.concatenate([[-np.inf], interior, [np.inf]])
            actual_counts, _ = np.histogram(vals, bins=full_edges)
            actual_pcts = np.clip(actual_counts / len(vals), 1e-6, None)
            actual_pcts = actual_pcts / actual_pcts.sum()

            expected_pcts = np.array(dist.expected_pcts, dtype=float)
            if len(expected_pcts) != len(actual_pcts):
                m = min(len(expected_pcts), len(actual_pcts))
                expected_pcts = expected_pcts[:m]
                actual_pcts = actual_pcts[:m]

            expected_pcts = np.clip(expected_pcts, 1e-6, None)
            expected_pcts = expected_pcts / expected_pcts.sum()

            # Compute PSI against reference proportions from training data.
            psi = float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))
            status = _psi_status(psi)
            actual_mean = float(np.mean(vals))
            mean_shift = actual_mean - dist.mean

            alert_msg = None
            if feat in CONDITION_FEATURES and status == "RED":
                alert_msg = f"{feat} PSI={psi:.3f} exceeds red threshold {PSI_RED:.2f}"
                alerts.append(alert_msg)

            feature_results.append(
                FeatureDriftResult(
                    feature=feat,
                    psi=psi,
                    status=status,
                    n_actual=int(len(vals)),
                    train_mean=dist.mean,
                    actual_mean=actual_mean,
                    mean_shift=mean_shift,
                    alert_message=alert_msg,
                )
            )

        # Overall status only considers condition features.
        cond_statuses = [r.status for r in feature_results if r.feature in CONDITION_FEATURES and r.status != "UNKNOWN"]
        if "RED" in cond_statuses:
            overall = "RED"
        elif "AMBER" in cond_statuses:
            overall = "AMBER"
        elif cond_statuses:
            overall = "GREEN"
        else:
            overall = "UNKNOWN"

        n_rows = int(len(new_df))
        n_batteries = int(new_df["battery_id"].nunique()) if "battery_id" in new_df.columns else 1
        report = DriftReport(
            overall_status=overall,
            n_rows=n_rows,
            n_batteries=n_batteries,
            features=feature_results,
            alerts=alerts,
        )
        report.log(label=label)
        return report

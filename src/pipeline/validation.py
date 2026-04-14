from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import get_config


@dataclass
class ValidationReport:
    schema_ok: bool
    physics_ok: bool
    structure_ok: bool
    issues: List[str]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "schema_ok": self.schema_ok,
            "physics_ok": self.physics_ok,
            "structure_ok": self.structure_ok,
            "issues": self.issues,
            "metrics": self.metrics,
        }


def _missing_ratio(df: pd.DataFrame, cols: List[str]) -> float:
    if not cols:
        return 1.0
    return float(df[cols].isna().mean().mean())


def _monotonic_fraction(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    dx = np.diff(x)
    return float(np.mean(dx >= 0))


def validate_timeseries(
    df_ts: pd.DataFrame,
    required_cols: List[str],
    *,
    time_col: str = "time",
    voltage_cols: Tuple[str, ...] = ("voltage_measured", "voltage_load"),
    current_cols: Tuple[str, ...] = ("current_measured", "current_load"),
    temp_col: str = "temperature_measured",
    max_missing_ratio: float = 0.10,
) -> ValidationReport:

    issues: List[str] = []
    metrics: Dict[str, float] = {}
    cfg = get_config()

    # 1) Required columns present
    missing = [c for c in required_cols if c not in df_ts.columns]
    schema_ok = (len(missing) == 0)
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # If schema fails badly, still compute what we can
    present_required = [c for c in required_cols if c in df_ts.columns]

    # 2) Missing values ratio
    miss_ratio = _missing_ratio(df_ts, present_required)
    metrics["missing_ratio_required"] = miss_ratio
    if miss_ratio > max_missing_ratio:
        issues.append(f"Too many missing values in required cols: {miss_ratio:.3f} > {max_missing_ratio}")

    # 3) Time monotonicity
    structure_ok = True
    if time_col in df_ts.columns:
        t = pd.to_numeric(df_ts[time_col], errors="coerce").dropna().to_numpy()
        mono = _monotonic_fraction(t)
        metrics["time_monotonic_fraction"] = mono
        if mono < 0.95:
            structure_ok = False
            issues.append(f"Time not monotonic enough: {mono:.3f} < 0.95")
        if len(t) and np.min(t) < 0:
            structure_ok = False
            issues.append("Time contains negative values")
    else:
        structure_ok = False
        issues.append(f"Missing time column '{time_col}'")

    # 4) Physics range checks (config-driven)
    physics_ok = True

    def range_check(col: str, lo: float, hi: float, name: str):
        nonlocal physics_ok
        if col not in df_ts.columns:
            return
        x = pd.to_numeric(df_ts[col], errors="coerce").dropna().to_numpy()
        if len(x) == 0:
            physics_ok = False
            issues.append(f"{name} column '{col}' has no numeric values")
            return
        mn, mx = float(np.min(x)), float(np.max(x))
        metrics[f"{col}_min"] = mn
        metrics[f"{col}_max"] = mx
        if mn < lo or mx > hi:
            physics_ok = False
            issues.append(f"{name} outside expected range in '{col}': min={mn:.3f} max={mx:.3f} expected [{lo},{hi}]")

    s = cfg.schema
    # Voltage
    for vc in voltage_cols:
        range_check(vc, float(s.voltage_v.min), float(s.voltage_v.max), "Voltage")

    # Current magnitude (supports signed charge/discharge streams)
    for cc in current_cols:
        if cc not in df_ts.columns:
            continue
        x = pd.to_numeric(df_ts[cc], errors="coerce").dropna().to_numpy()
        if len(x) == 0:
            physics_ok = False
            issues.append(f"Current column '{cc}' has no numeric values")
            continue
        x_abs = np.abs(x)
        mn, mx = float(np.min(x_abs)), float(np.max(x_abs))
        metrics[f"{cc}_abs_min"] = mn
        metrics[f"{cc}_abs_max"] = mx
        if mn < float(s.current_a.min) or mx > float(s.current_a.max):
            physics_ok = False
            issues.append(
                f"Current magnitude outside expected range in '{cc}': "
                f"abs_min={mn:.3f} abs_max={mx:.3f} "
                f"expected [{s.current_a.min},{s.current_a.max}]"
            )

    # Temperature
    range_check(temp_col, float(s.temperature_c.min), float(s.temperature_c.max), "Temperature")

    return ValidationReport(
        schema_ok=schema_ok,
        physics_ok=physics_ok,
        structure_ok=structure_ok,
        issues=issues,
        metrics=metrics,
    )

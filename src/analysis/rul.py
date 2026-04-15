# src/analysis/rul.py
from __future__ import annotations

import pandas as pd
import numpy as np
from src.config import get_config


def build_cycle_table_from_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-cycle discharge table from metadata.

    Requires canonical columns after mapping:
        - battery_id, filename
        - type (optional but typical)
        - start_time or uid (optional)
        - capacity (or Capacity)
    """
    df = meta.copy()

    # normalize type
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        df = df[df["type"] == "discharge"].copy()

    if "battery_id" not in df.columns:
        raise ValueError("Missing 'battery_id' in metadata after mapping.")
    if "filename" not in df.columns:
        raise ValueError("Missing 'filename' in metadata after mapping.")

    # capacity normalization
    if "capacity" not in df.columns:
        if "Capacity" in df.columns:
            df = df.rename(columns={"Capacity": "capacity"})
        else:
            df["capacity"] = np.nan

    # Sort preference: start_time > uid > filename
    sort_cols = []
    if "start_time" in df.columns:
        sort_cols.append("start_time")
    if "uid" in df.columns:
        sort_cols.append("uid")
    sort_cols.append("filename")

    df = df.sort_values(["battery_id"] + sort_cols, kind="mergesort").copy()
    df["cycle_index"] = df.groupby("battery_id").cumcount() + 1

    return df[["battery_id", "cycle_index", "filename", "capacity"]].copy()


def _robust_initial_capacity(capacity_series: pd.Series) -> float:
    """
    Estimate a battery's true initial (healthy) capacity robustly.

    Strategy:
      1. Convert to numeric and drop NaNs.
      2. Keep only values above the configured schema minimum (default 1.0 Ah)
         so that deep-discharge test pulses (e.g. 0.5 Ah) are ignored.
      3. Take the median of the first up to 10 such values.
      4. If no valid values exist in the first 10 cycles, try the full series.
      5. If still nothing, fall back to the configured nominal capacity (2.0 Ah).

    This fixes the bug where batteries like B0033 that have early deep-discharge
    test pulses (capacity ~0.5 Ah) had their initial capacity set to 0.5 Ah,
    causing a cascading wrong EOL threshold and wrong RUL throughout.
    """
    cfg = get_config()
    nominal = float(cfg.eol.nominal_capacity_ahr)   # 2.0 Ah from pipeline.yaml
    schema_min = 1.0  # minimum physically valid capacity (Ah)

    cap = pd.to_numeric(capacity_series, errors="coerce")

    # --- Step 1: try first 10 cycles, values above schema_min ---
    first_10 = cap.iloc[:10]
    valid_first_10 = first_10[first_10 > schema_min].dropna()
    if len(valid_first_10) >= 1:
        return float(np.median(valid_first_10.values))

    # --- Step 2: try the full series ---
    valid_all = cap[cap > schema_min].dropna()
    if len(valid_all) >= 1:
        return float(np.median(valid_all.values))

    # --- Step 3: absolute fallback to nominal ---
    return nominal


def add_rul(cycle_table: pd.DataFrame, alpha: float | None = None) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for every cycle of every battery.

    EOL threshold = alpha * initial_capacity
    RUL(cycle)    = max(eol_cycle - cycle_index, 0)

    The initial capacity is estimated robustly using the median of the first
    10 physically valid capacity readings (above 1.0 Ah), so that early
    deep-discharge test pulses do not corrupt the threshold.
    """
    if alpha is None:
        alpha = float(get_config().eol.capacity_fade_fraction)

    df = cycle_table.copy()
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    def per_battery(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("cycle_index").copy()

        # --- FIX: robust initial capacity (was: first finite value) ---
        init_cap = _robust_initial_capacity(g["capacity"])

        if not np.isfinite(init_cap) or init_cap <= 0:
            g["init_capacity"] = np.nan
            g["eol_capacity_threshold"] = np.nan
            g["eol_cycle"] = int(g["cycle_index"].max())
            g["RUL"] = 0
            return g

        thr = alpha * init_cap

        # Any cycle with capacity below 50% of init_cap is a test pulse /
        # impedance measurement, not a real discharge. Skip these when
        # detecting EOL crossing so that early test pulses (e.g. B0033
        # cycle 1 = 0.69 Ah) do not falsely trigger EOL at cycle 1.
        pulse_floor = 0.5 * init_cap

        # Find the FIRST cycle where capacity permanently crosses EOL threshold.
        eol_cycle = None
        for cyc, cap in zip(g["cycle_index"].values, g["capacity"].values):
            if np.isfinite(cap) and cap > pulse_floor and cap <= thr:
                eol_cycle = int(cyc)
                break

        if eol_cycle is None:
            eol_cycle = int(g["cycle_index"].max())

        g["init_capacity"] = init_cap
        g["eol_capacity_threshold"] = thr
        g["eol_cycle"] = eol_cycle
        g["RUL"] = (eol_cycle - g["cycle_index"]).clip(lower=0).astype(int)
        return g

    groups = [per_battery(g) for _, g in df.groupby("battery_id", sort=False)]
    return pd.concat(groups, axis=0) if groups else df.copy()

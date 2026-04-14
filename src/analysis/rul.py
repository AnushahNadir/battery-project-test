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


def add_rul(cycle_table: pd.DataFrame, alpha: float | None = None) -> pd.DataFrame:
    """
    EOL threshold = alpha * initial capacity
    RUL(cycle) = max(eol_cycle - cycle_index, 0)
    """
    if alpha is None:
        alpha = float(get_config().eol.capacity_fade_fraction)

    df = cycle_table.copy()
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    def per_battery(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("cycle_index").copy()

        # initial capacity = first finite
        init_cap = np.nan
        for v in g["capacity"].values:
            if np.isfinite(v):
                init_cap = v
                break

        if not np.isfinite(init_cap):
            g["init_capacity"] = np.nan
            g["eol_capacity_threshold"] = np.nan
            g["eol_cycle"] = int(g["cycle_index"].max())
            g["RUL"] = 0
            return g

        thr = alpha * init_cap

        eol_cycle = None
        for cyc, cap in zip(g["cycle_index"].values, g["capacity"].values):
            if np.isfinite(cap) and cap <= thr:
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

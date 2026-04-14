# src/pipeline/mapper.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.schema import SYNONYMS_META, SYNONYMS_TS

logger = logging.getLogger(__name__)


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())


# -----------------------------
# Stage 1: deterministic value hints (ONLY used as fallback)
# Canonical names MUST match schema.py (SYNONYMS_TS keys)
# -----------------------------
CANON_TS_REQUIRED: List[str] = [
    "time",
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "current_load",
    "voltage_load",
]


@dataclass
class HintScore:
    field: str        # canonical field name (schema key)
    column: str       # raw column name in df
    score: float
    reasons: List[str]


def _to_numeric_series(s: pd.Series) -> Optional[pd.Series]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return x if len(x) >= 10 else None


def _monotonic_fraction(a: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    return float(np.mean(np.diff(a) >= 0))


def _near_zero_fraction(a: np.ndarray, thr: float = 0.5) -> float:
    return float(np.mean(np.abs(a) < thr))


def _unique_ratio(a: np.ndarray, decimals: int = 4) -> float:
    ar = np.round(a, decimals=decimals)
    return float(len(np.unique(ar)) / len(ar))


def _median_abs_diff(a: np.ndarray) -> float:
    if len(a) < 3:
        return float("inf")
    return float(np.median(np.abs(np.diff(a))))


# ---- scorers (return HintScore with schema canonical field name) ----

def score_time(col: str, s: pd.Series) -> HintScore:
    x = _to_numeric_series(s)
    if x is None:
        return HintScore("time", col, 0.0, ["too few numeric values"])
    a = x.to_numpy()

    mono = _monotonic_fraction(a)
    nonneg = float(np.min(a) >= 0)
    start0 = float(np.min(a) <= 1.0)
    span_val = float(np.max(a) - np.min(a))
    span_ok = float(span_val > 30.0)  # tighten: time should span "real" seconds/minutes

    score = 0.0
    reasons: List[str] = []
    score += 3.5 * mono; reasons.append(f"monotonic_fraction={mono:.3f}")
    score += 1.0 * nonneg; reasons.append(f"nonneg={bool(nonneg)}")
    score += 0.7 * start0; reasons.append(f"starts_near_zero={bool(start0)}")
    score += 2.0 * span_ok; reasons.append(f"span={span_val:.3f} span_ok(>30)={bool(span_ok)}")

    # penalize tiny-span monotonic signals (temperature often drifts monotonically)
    if span_val < 5.0:
        score -= 1.5
        reasons.append("penalty: tiny span (<5) likely not time")

    return HintScore("time", col, float(score), reasons)


def score_temperature(col: str, s: pd.Series) -> HintScore:
    x = _to_numeric_series(s)
    if x is None:
        return HintScore("temperature_measured", col, 0.0, ["too few numeric values"])
    a = x.to_numpy()

    mn, mx = float(np.min(a)), float(np.max(a))
    std = float(np.std(a))
    mad = _median_abs_diff(a)
    mono = _monotonic_fraction(a)

    in_phys = float(mn >= -10 and mx <= 120)
    smooth = float(mad < 0.5)
    low_var = float(std < 20.0)
    not_time = float(mono < 0.995)  # temperature can be monotonic; keep soft

    score = 0.0
    reasons: List[str] = []
    score += 2.0 * in_phys; reasons.append(f"in_range[-10,120]={bool(in_phys)} mn={mn:.2f} mx={mx:.2f}")
    score += 1.4 * smooth; reasons.append(f"median_abs_diff={mad:.4f}")
    score += 1.0 * low_var; reasons.append(f"std={std:.3f}")
    score += 0.3 * not_time; reasons.append(f"not_time_like={bool(not_time)}")

    # penalize voltage-looking
    if 2.0 <= mn and mx <= 5.5:
        score -= 0.7
        reasons.append("penalty: voltage-like range")

    return HintScore("temperature_measured", col, float(score), reasons)


def score_current_measured(col: str, s: pd.Series) -> HintScore:
    x = _to_numeric_series(s)
    if x is None:
        return HintScore("current_measured", col, 0.0, ["too few numeric values"])
    a = x.to_numpy()

    mn, mx = float(np.min(a)), float(np.max(a))
    has_neg = float(mn < -0.05)
    has_pos = float(mx > 0.05)
    signed = float(has_neg and has_pos)
    ampish = float((abs(mn) > 0.2) or (abs(mx) > 0.2))
    ur = _unique_ratio(a, decimals=4)

    score = 0.0
    reasons: List[str] = []
    score += 2.2 * has_neg; reasons.append(f"has_neg={bool(has_neg)} mn={mn:.3f}")
    score += 0.9 * has_pos; reasons.append(f"has_pos={bool(has_pos)} mx={mx:.3f}")
    score += 0.8 * signed;  reasons.append(f"signed={bool(signed)}")
    score += 0.8 * ampish;  reasons.append(f"ampish={bool(ampish)}")
    score += 0.6 * float(ur > 0.10); reasons.append(f"unique_ratio={ur:.3f} (prefer higher)")

    # penalize voltage-like / time-like
    if mn >= 2.0 and mx <= 5.5:
        score -= 1.2; reasons.append("penalty: voltage-like range")
    if _monotonic_fraction(a) > 0.995 and mn >= 0:
        score -= 0.8; reasons.append("penalty: time-like")

    return HintScore("current_measured", col, float(score), reasons)


def score_current_load(col: str, s: pd.Series) -> HintScore:
    x = _to_numeric_series(s)
    if x is None:
        return HintScore("current_load", col, 0.0, ["too few numeric values"])
    a = x.to_numpy()

    mn, mx = float(np.min(a)), float(np.max(a))
    ur = _unique_ratio(a, decimals=4)
    step_like = float(ur < 0.05)
    amp_range = float(mn >= -5 and mx <= 5)

    score = 0.0
    reasons: List[str] = []
    score += 2.3 * step_like; reasons.append(f"step_like={bool(step_like)} unique_ratio={ur:.3f}")
    score += 0.8 * amp_range; reasons.append(f"amp_range[-5,5]={bool(amp_range)}")

    if ur > 0.20:
        score -= 1.0; reasons.append("penalty: too continuous for load current")

    return HintScore("current_load", col, float(score), reasons)


def _score_voltage_base(col: str, s: pd.Series) -> Tuple[float, List[str], Optional[np.ndarray]]:
    x = _to_numeric_series(s)
    if x is None:
        return 0.0, ["too few numeric values"], None
    a = x.to_numpy()

    mn, mx = float(np.min(a)), float(np.max(a))
    in_v = float(mn >= 2.0 and mx <= 5.5)
    not_time = float(_monotonic_fraction(a) < 0.995)

    score = 0.0
    reasons: List[str] = []
    score += 2.0 * in_v; reasons.append(f"in_range[2,5.5]={bool(in_v)} mn={mn:.3f} mx={mx:.3f}")
    score += 0.4 * not_time; reasons.append(f"not_time_like={bool(not_time)}")
    return float(score), reasons, a


def score_voltage_measured(col: str, s: pd.Series) -> HintScore:
    base, reasons, a = _score_voltage_base(col, s)
    if a is None:
        return HintScore("voltage_measured", col, 0.0, reasons)

    nz = _near_zero_fraction(a, thr=0.5)
    mad = _median_abs_diff(a)

    score = base
    score += 0.6 * float(nz < 0.01); reasons.append(f"near_zero_fraction={nz:.3f} (prefer low)")
    score += 0.4 * float(mad < 0.2); reasons.append(f"median_abs_diff={mad:.4f} (prefer smooth)")
    return HintScore("voltage_measured", col, float(score), reasons)


def score_voltage_load(col: str, s: pd.Series) -> HintScore:
    base, reasons, a = _score_voltage_base(col, s)
    if a is None:
        return HintScore("voltage_load", col, 0.0, reasons)

    nz = _near_zero_fraction(a, thr=0.5)
    score = base
    score += 1.2 * float(nz > 0.02); reasons.append(f"near_zero_fraction={nz:.3f} (prefer higher)")
    return HintScore("voltage_load", col, float(score), reasons)


def suggest_ts_mapping_from_values(df: pd.DataFrame) -> Tuple[Dict[str, str], List[HintScore]]:
    scorers = [
        score_time,
        score_temperature,
        score_current_measured,
        score_current_load,
        score_voltage_measured,
        score_voltage_load,
    ]

    all_scores: List[HintScore] = []
    for col in df.columns:
        for scorer in scorers:
            all_scores.append(scorer(col, df[col]))

    # sort by score descending
    all_scores_sorted = sorted(all_scores, key=lambda hs: hs.score, reverse=True)

    # greedy assignment: field -> best available column
    mapping: Dict[str, str] = {}
    used_cols = set()

    for hs in all_scores_sorted:
        if hs.score <= 0:
            continue
        if hs.field in mapping:
            continue
        if hs.column in used_cols:
            continue
        mapping[hs.field] = hs.column
        used_cols.add(hs.column)
        if len(mapping) == len(CANON_TS_REQUIRED):
            break

    return mapping, all_scores_sorted


# -----------------------------
# Mapper (synonyms first, value hints second)
# -----------------------------

def build_rename_map(
    df: pd.DataFrame,
    kind: str = "meta",
    interactive: bool = True,
    verbose: bool = False,
) -> Dict[str, str]:

    """
    Maps raw columns -> canonical columns based on synonyms.
    kind: "meta" or "ts"
    """
    synonyms = SYNONYMS_META if kind == "meta" else SYNONYMS_TS
    raw_cols = list(df.columns)
    raw_norm = {_normalize_name(c): c for c in raw_cols}

    rename: Dict[str, str] = {}
    used_raw = set()

    # 1) synonym mapping (stable, deterministic)
    for canon, candidates in synonyms.items():
        found = None
        for cand in candidates:
            cand_norm = _normalize_name(cand)
            if cand_norm in raw_norm:
                found = raw_norm[cand_norm]
                break
        if found:
            rename[found] = canon
            used_raw.add(found)

    # 2) value-based fallback for TS only (fills missing canonical fields)
    if kind == "ts":
        already = set(rename.values())
        missing = [c for c in CANON_TS_REQUIRED if c not in already]

        if missing:
            hinted, scores = suggest_ts_mapping_from_values(df)

            # show candidates clearly (not misleading global top list)
            if verbose:
                logger.info("\n[Mapper] Value-hint candidates (top 3 per field):")
            for field in CANON_TS_REQUIRED:
                ...

            for field in CANON_TS_REQUIRED:
                best = [s for s in scores if s.field == field and s.score > 0][:3]
                for s in best:
                    logger.info(f"  - {field:20s} <- {s.column:22s} score={s.score:.3f}")

            logger.info("\n[Mapper] Value-hint chosen mapping (only for missing fields):")
            for field in missing:
                if field in hinted:
                    logger.info(f"  - {field:20s} <- {hinted[field]}")

            # apply only missing fields, never override synonym matches
            for canon_field in missing:
                raw_col = hinted.get(canon_field)
                if not raw_col:
                    continue
                if raw_col not in rename:  # don't overwrite raw->canon mapping
                    rename[raw_col] = canon_field
                    used_raw.add(raw_col)

    # 3) interactive mapping (restricted to canonical names only)
    unknown = [c for c in raw_cols if c not in used_raw]
    if unknown and interactive:
        allowed = set(synonyms.keys())
        logger.info(f"\n[Mapper] Unmapped columns detected ({kind}):")
        for c in unknown:
            logger.info(f" - {c}")
        logger.info("\nMap them to a CANONICAL name, or press ENTER to skip.")
        logger.info(f"Allowed canonical names: {sorted(allowed)}")

        for c in unknown:
            ans = input(f"Map '{c}' -> ").strip()
            if ans:
                if ans not in allowed:
                    logger.info(f"[Mapper] Rejected: '{ans}' is not a known canonical column. Skipping.")
                    continue
                rename[c] = ans
                used_raw.add(c)

    return rename


def standardize_columns(
    df: pd.DataFrame,
    kind: str = "meta",
    interactive: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    rename = build_rename_map(
        df,
        kind=kind,
        interactive=interactive,
        verbose=verbose,
    )
    return df.rename(columns=rename).copy()


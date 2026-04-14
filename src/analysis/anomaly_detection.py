# src/analysis/anomaly_detection.py
"""
Anomaly Detection (Deterministic, Non-LLM)

Purpose:
- Detect suspicious / outlier rows in the engineered cycle-level dataset
  (e.g., cycle_features_with_rul.csv).
- Produce an auditable report (JSON + Markdown) that can be used for:
  - QC before modeling
  - Debugging mapping / feature extraction issues
  - Flagging batteries/cycles that look physically or statistically odd

Hard rules:
- Deterministic (no randomness).
- No "smart guesses" about cause. Only signal-based flags.
- Produces evidence: thresholds, counts, example rows.

Recommended input columns (if available):
  battery_id, cycle_index, RUL, capacity,
  temp_mean, temp_max, v_min, v_mean, i_mean, i_min, energy_j, ah_est, duration_s
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "capacity",
    "temp_mean",
    "temp_max",
    "v_min",
    "v_mean",
    "i_mean",
    "i_min",
    "energy_j",
    "ah_est",
    "duration_s",
]


@dataclass
class AnomalyFlag:
    """One anomaly finding."""
    rule_id: str
    severity: str  # "INFO" | "WARNING" | "ERROR"
    feature: str
    message: str
    n_flagged: int
    threshold: str
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "feature": self.feature,
            "message": self.message,
            "n_flagged": int(self.n_flagged),
            "threshold": self.threshold,
            "examples": self.examples,
        }


@dataclass
class AnomalyReport:
    """Full anomaly report."""
    total_rows: int
    total_flagged_rows: int
    flagged_row_ratio: float
    rules_triggered: int
    flags: List[AnomalyFlag]
    flagged_rows_sample: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": int(self.total_rows),
            "total_flagged_rows": int(self.total_flagged_rows),
            "flagged_row_ratio": float(self.flagged_row_ratio),
            "rules_triggered": int(self.rules_triggered),
            "flags": [f.to_dict() for f in self.flags],
            "flagged_rows_sample": self.flagged_rows_sample,
        }


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Robust z-score using median and MAD (median absolute deviation).
    z = 0.6745 * (x - median) / MAD
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < eps:
        return np.full_like(x, np.nan, dtype=float)
    return 0.6745 * (x - med) / mad


def _examples_from_df(df: pd.DataFrame, idxs: np.ndarray, cols: List[str], k: int = 5) -> List[Dict[str, Any]]:
    if idxs.size == 0:
        return []
    take = idxs[: min(k, idxs.size)]
    return df.loc[take, cols].to_dict(orient="records")


def detect_anomalies(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    *,
    id_cols: Optional[List[str]] = None,
    robust_z_threshold: float = 6.0,
    iqr_k: float = 3.0,
    max_missing_ratio: float = 0.5,
    max_samples_in_examples: int = 5,
) -> AnomalyReport:
    """
    Detect anomalies using simple, deterministic rules:
    - Robust z-score outliers per feature (MAD-based)
    - IQR outliers per feature (Tukey fences)
    - Excessive missingness per feature
    - Basic sanity checks on key columns if present (duration_s >= 0, etc.)

    Returns an AnomalyReport.

    Notes:
    - This does NOT reject datasets by itself. It only reports.
    - You can decide gating logic elsewhere (e.g., "if too many ERROR flags, reject").
    """
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    id_cols = id_cols or [c for c in ["battery_id", "cycle_index", "filename"] if c in df.columns]

    # Only keep columns that exist
    feat = [c for c in feature_columns if c in df.columns]
    if not feat:
        return AnomalyReport(
            total_rows=len(df),
            total_flagged_rows=0,
            flagged_row_ratio=0.0,
            rules_triggered=0,
            flags=[
                AnomalyFlag(
                    rule_id="A000",
                    severity="WARNING",
                    feature="(none)",
                    message="No known feature columns found for anomaly detection",
                    n_flagged=0,
                    threshold="N/A",
                    examples=[],
                )
            ],
            flagged_rows_sample=[],
        )

    work = df.copy()

    # Coerce features to numeric
    for c in feat:
        work[c] = _safe_numeric(work[c])

    flags: List[AnomalyFlag] = []
    flagged_mask = np.zeros(len(work), dtype=bool)

    example_cols = id_cols + feat
    example_cols = [c for c in example_cols if c in work.columns]

    # Rule A001: Missingness
    for c in feat:
        miss_ratio = float(work[c].isna().mean())
        if miss_ratio > max_missing_ratio:
            # This flags the *feature* as problematic rather than specific rows
            flags.append(
                AnomalyFlag(
                    rule_id="A001",
                    severity="WARNING",
                    feature=c,
                    message=f"High missingness for feature '{c}'",
                    n_flagged=int(work[c].isna().sum()),
                    threshold=f"missing_ratio <= {max_missing_ratio:.2f}",
                    examples=_examples_from_df(work, work.index[work[c].isna()].to_numpy(), example_cols, k=max_samples_in_examples),
                )
            )

    # Rule A002: Robust z-score outliers
    for c in feat:
        x = work[c].to_numpy(dtype=float)
        rz = _robust_zscore(x)
        if np.all(np.isnan(rz)):
            continue
        idxs = np.where(np.abs(rz) > robust_z_threshold)[0]
        if idxs.size > 0:
            flagged_mask[idxs] = True
            flags.append(
                AnomalyFlag(
                    rule_id="A002",
                    severity="WARNING",
                    feature=c,
                    message=f"Robust z-score outliers detected for '{c}'",
                    n_flagged=int(idxs.size),
                    threshold=f"|robust_z| <= {robust_z_threshold:.1f}",
                    examples=_examples_from_df(work.reset_index(drop=True), idxs, example_cols, k=max_samples_in_examples),
                )
            )

    # Rule A003: IQR outliers (Tukey fences)
    for c in feat:
        s = work[c].dropna()
        if len(s) < 20:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr

        x = work[c].to_numpy(dtype=float)
        idxs = np.where((x < lo) | (x > hi))[0]
        if idxs.size > 0:
            flagged_mask[idxs] = True
            flags.append(
                AnomalyFlag(
                    rule_id="A003",
                    severity="INFO",
                    feature=c,
                    message=f"IQR outliers detected for '{c}'",
                    n_flagged=int(idxs.size),
                    threshold=f"[{lo:.4g}, {hi:.4g}] (k={iqr_k:.1f})",
                    examples=_examples_from_df(work.reset_index(drop=True), idxs, example_cols, k=max_samples_in_examples),
                )
            )

    # Rule A010: Basic sanity checks on common columns (if present)
    # These are simple and safe; they catch obvious pipeline bugs.
    sanity_rules: List[Tuple[str, str, str, str]] = [
        ("A010", "ERROR", "duration_s", "duration_s should be >= 0"),
        ("A011", "ERROR", "RUL", "RUL should be >= 0"),
        ("A012", "WARNING", "capacity", "capacity should be > 0 (non-positive is suspicious)"),
        ("A013", "WARNING", "ah_est", "ah_est should be finite (NaN/inf is suspicious if TS was found)"),
    ]

    for rule_id, severity, col, msg in sanity_rules:
        if col not in work.columns:
            continue
        x = work[col].to_numpy(dtype=float)
        if rule_id in {"A010", "A011"}:
            idxs = np.where(np.isfinite(x) & (x < 0))[0]
        elif rule_id == "A012":
            idxs = np.where(np.isfinite(x) & (x <= 0))[0]
        else:
            idxs = np.where(~np.isfinite(x))[0]

        if idxs.size > 0:
            flagged_mask[idxs] = True if severity in {"ERROR", "WARNING"} else flagged_mask[idxs]
            flags.append(
                AnomalyFlag(
                    rule_id=rule_id,
                    severity=severity,
                    feature=col,
                    message=msg,
                    n_flagged=int(idxs.size),
                    threshold="see message",
                    examples=_examples_from_df(work.reset_index(drop=True), idxs, example_cols, k=max_samples_in_examples),
                )
            )

    total_rows = int(len(work))
    total_flagged = int(flagged_mask.sum())
    ratio = float(total_flagged / total_rows) if total_rows else 0.0

    # Small sample of flagged rows for quick debugging
    flagged_idxs = np.where(flagged_mask)[0]
    flagged_rows_sample = _examples_from_df(
        work.reset_index(drop=True),
        flagged_idxs,
        cols=example_cols,
        k=max_samples_in_examples,
    )

    return AnomalyReport(
        total_rows=total_rows,
        total_flagged_rows=total_flagged,
        flagged_row_ratio=ratio,
        rules_triggered=sum(1 for f in flags if f.n_flagged > 0),
        flags=flags,
        flagged_rows_sample=flagged_rows_sample,
    )


def generate_anomaly_report_md(report: AnomalyReport) -> str:
    """Render the anomaly report to Markdown."""
    lines: List[str] = []
    lines.append("# Anomaly Detection Report")
    lines.append("")
    lines.append("This report flags unusual rows/features using deterministic rules (no ML, no LLM).")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total rows: **{report.total_rows}**")
    lines.append(f"- Flagged rows: **{report.total_flagged_rows}** ({report.flagged_row_ratio*100:.1f}%)")
    lines.append(f"- Rules triggered: **{report.rules_triggered}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Flags")
    lines.append("")
    if not report.flags:
        lines.append("*No flags generated.*")
        return "\n".join(lines)

    lines.append("| Rule | Severity | Feature | # Flagged | Threshold | Message |")
    lines.append("|------|----------|---------|-----------|-----------|---------|")
    for f in report.flags:
        lines.append(
            f"| {f.rule_id} | {f.severity} | `{f.feature}` | {f.n_flagged} | {f.threshold} | {f.message} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Flagged Rows Sample")
    lines.append("")
    if report.flagged_rows_sample:
        # Print as JSON-ish block to avoid massive tables
        lines.append("```json")
        lines.append(json.dumps(report.flagged_rows_sample, indent=2)[:4000])  # protect against huge output
        lines.append("```")
    else:
        lines.append("*No flagged row samples available.*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Flags are **signals**, not proof of error.")
    lines.append("- If many ERROR/WARNING flags appear, it often means mapping or feature extraction issues.")
    lines.append("- Use this report to decide whether to re-check Stage-1/2 outputs before modeling.")
    return "\n".join(lines)


def run_anomaly_detection(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    feature_columns: Optional[List[str]] = None,
) -> AnomalyReport:
    """
    Convenience runner:
    - Loads CSV
    - Detects anomalies
    - Writes:
        anomaly_report.json
        anomaly_report.md
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    report = detect_anomalies(df, feature_columns=feature_columns)

    (output_dir / "anomaly_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "anomaly_report.md").write_text(
        generate_anomaly_report_md(report), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    # Example:
    # python -m src.analysis.anomaly_detection
    base = Path("d:/Energy Project/code/battery-project3")
    inp = base / "data/processed/cycle_features_with_rul.csv"
    out = base / "data/processed/modeling"

    if inp.exists():
        rep = run_anomaly_detection(inp, out)
        logger.info("Anomaly report written:")
        logger.info(f"- {out / 'anomaly_report.json'}")
        logger.info(f"- {out / 'anomaly_report.md'}")
        logger.info(f"Flagged rows: {rep.total_flagged_rows}/{rep.total_rows} ({rep.flagged_row_ratio*100:.1f}%)")
    else:
        logger.info(f"Input not found: {inp}")

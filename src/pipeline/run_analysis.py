# src/pipeline/run_analysis.py
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.pipeline.data_loader import load_metadata, save_csv
from src.pipeline.mapper import build_rename_map, standardize_columns
from src.pipeline.validation import validate_timeseries
from src.pipeline.confidence import compute_confidence
from src.pipeline.drift import DriftMonitor
from src.pipeline.errors import DatasetRejectedError
from src.config import get_config

from src.analysis.rul import build_cycle_table_from_metadata, add_rul
from src.analysis.degradation_features import build_timeseries_features


# --- Stage 4 configuration ---
REQUIRED_TS = [
    "time",
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "current_load",
    "voltage_load",
]

_CFG = get_config()
LOW_THRESHOLD = float(_CFG.gating.low_threshold)
REVIEW_THRESHOLD = float(_CFG.gating.review_threshold)


def run(
    metadata_path: str | Path,
    raw_root: str | Path,
    out_dir: str | Path,
    alpha: float,
    non_interactive: bool,
):
    out_dir = Path(out_dir)
    raw_root = Path(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load metadata
    meta = load_metadata(metadata_path)

    # 2) Standardize metadata
    rename_map = build_rename_map(meta, kind="meta", interactive=(not non_interactive))
    (out_dir / "mapping_used.json").write_text(json.dumps(rename_map, indent=2), encoding="utf-8")

    meta_std = meta.rename(columns=rename_map).copy()
    save_csv(meta_std, out_dir / "metadata_standardized.csv")

    # 3) Build discharge cycle table + compute RUL (metadata only)
    cycle_table = build_cycle_table_from_metadata(meta_std)
    cycle_rul = add_rul(cycle_table, alpha=alpha)
    save_csv(cycle_rul, out_dir / "cycle_table_with_rul.csv")

    # ---- STAGE 4: Gating BEFORE full feature extraction ----
    sample_row = cycle_rul[["battery_id", "cycle_index", "filename"]].dropna().head(1)
    if sample_row.empty:
        raise DatasetRejectedError("No filenames found in cycle_rul. Check metadata 'filename' column mapping.")

    sample = sample_row.iloc[0]
    filename = str(sample["filename"])
    battery_id = str(sample["battery_id"])

    candidate_paths = [
        raw_root / filename,
        raw_root / battery_id / filename,
    ]
    ts_path = next((p for p in candidate_paths if p.exists()), None)

    # recursive fallback
    if ts_path is None:
        matches = list(raw_root.rglob(filename))
        if matches:
            ts_path = matches[0]

    if ts_path is None:
        raise DatasetRejectedError(
            f"Could not find a representative TS file for validation. Tried: {[str(p) for p in candidate_paths]}"
        )

    df_ts_raw = pd.read_csv(ts_path)
    df_ts_std = standardize_columns(df_ts_raw, kind="ts", interactive=(not non_interactive))

    validation = validate_timeseries(df_ts_std, required_cols=REQUIRED_TS)
    schema_match_ratio = sum(c in df_ts_std.columns for c in REQUIRED_TS) / len(REQUIRED_TS)

    # simple placeholder: if schema_ok assume decent agreement
    pattern_agreement_ratio = 0.75 if validation.schema_ok else 0.40

    conf = compute_confidence(
        schema_match_ratio=schema_match_ratio,
        pattern_agreement_ratio=pattern_agreement_ratio,
        validation=validation,
        imputation_fraction=float(validation.metrics.get("missing_ratio_required", 0.0)),
    )

    (out_dir / "validation_report.json").write_text(json.dumps(validation.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "confidence_report.json").write_text(json.dumps(conf.to_dict(), indent=2), encoding="utf-8")

    if conf.confidence < LOW_THRESHOLD:
        raise DatasetRejectedError(f"Dataset rejected. Confidence={conf.confidence:.2f}. Issues={validation.issues}")

    trigger_rag_assist = conf.confidence < REVIEW_THRESHOLD
    (out_dir / "trigger_rag_assist.txt").write_text(str(trigger_rag_assist), encoding="utf-8")

    # 4) Extract time-series engineered features
    ts_feat = build_timeseries_features(
        cycle_rul[["battery_id", "cycle_index", "filename"]],
        raw_root=raw_root,
    )
    save_csv(ts_feat, out_dir / "time_series_features.csv")

    # 5) Merge into one supervised table
    merged = cycle_rul.merge(ts_feat, on=["battery_id", "cycle_index", "filename"], how="left")
    save_csv(merged, out_dir / "cycle_features_with_rul.csv")

    # ---- QC checks (this is what makes preprocessing "done") ----
    total = len(merged)
    found = int(merged["ts_found"].sum()) if "ts_found" in merged.columns else 0
    missing = total - found

    # Hard check 1: too many TS missing
    missing_rate = (missing / total) if total else 1.0
    if missing_rate > 0.20:
        raise DatasetRejectedError(
            f"Too many missing time-series files: {missing}/{total} ({missing_rate:.1%}). "
            f"Check raw_root and filename mapping."
        )

    # Hard check 2: if TS exists, RUL must exist
    if "ts_found" in merged.columns and "RUL" in merged.columns:
        bad = merged[(merged["ts_found"] == True) & (merged["RUL"].isna())]
        if not bad.empty:
            raise DatasetRejectedError(f"{len(bad)} rows have ts_found=True but missing RUL.")

    # Write QC report
    qc = {
        "rows_total": total,
        "ts_found": found,
        "ts_missing": missing,
        "ts_missing_rate": missing_rate,
        "columns": list(merged.columns),
        "rul_missing": int(merged["RUL"].isna().sum()) if "RUL" in merged.columns else None,
    }

    # Optional drift check against trained reference distributions.
    # This activates automatically when trained_models/feature_distributions.json exists.
    drift_info: dict[str, object]
    try:
        base_dir = Path(__file__).resolve().parents[2]
        ref_path = base_dir / _CFG.paths.trained_models / "feature_distributions.json"
        if ref_path.exists():
            monitor = DriftMonitor.load(ref_path)
            drift_report = monitor.compute(merged, label="incoming_batch")
            drift_path = out_dir / "drift_report.json"
            drift_path.write_text(json.dumps(drift_report.to_dict(), indent=2), encoding="utf-8")
            drift_info = {
                "status": "OK",
                "reference_path": str(ref_path),
                "report_path": str(drift_path),
                "overall_status": drift_report.overall_status,
                "n_alerts": len(drift_report.alerts),
                "alerts": drift_report.alerts,
            }
        else:
            drift_info = {
                "status": "SKIPPED",
                "reason": f"reference distribution not found at {ref_path}",
            }
    except Exception as exc:
        drift_info = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
    qc["drift"] = drift_info

    (out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    return {
        "metadata_standardized": str(out_dir / "metadata_standardized.csv"),
        "cycle_table_with_rul": str(out_dir / "cycle_table_with_rul.csv"),
        "time_series_features": str(out_dir / "time_series_features.csv"),
        "cycle_features_with_rul": str(out_dir / "cycle_features_with_rul.csv"),
        "mapping_used": str(out_dir / "mapping_used.json"),
        "validation_report": str(out_dir / "validation_report.json"),
        "confidence_report": str(out_dir / "confidence_report.json"),
        "trigger_rag_assist": str(out_dir / "trigger_rag_assist.txt"),
        "qc_report": str(out_dir / "qc_report.json"),
    }

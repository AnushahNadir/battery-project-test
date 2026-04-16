# src/verification/check_group_split.py
# Updated: 2026-04-16
"""
Verifies that the battery-level train/cal/test split has zero leakage.

Run before every pipeline execution:
    python -m src.verification.check_group_split

Exits with code 1 if any overlap is detected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]


def _load_split_metadata(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        print(f"[ERROR] model_metrics.json not found at {metrics_path}")
        print("        Run the pipeline at least once before checking splits.")
        sys.exit(1)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta = data.get("split_metadata", {})
    if not meta:
        print("[ERROR] No split_metadata found in model_metrics.json")
        sys.exit(1)
    return meta


def _load_features(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[ERROR] Feature CSV not found: {csv_path}")
        sys.exit(1)
    return pd.read_csv(csv_path, usecols=["battery_id"])


def check_split(meta: dict, df: pd.DataFrame) -> bool:
    train     = set(str(b) for b in meta.get("actual_train_batteries", []))
    cal       = set(str(b) for b in meta.get("cal_batteries", []))
    test      = set(str(b) for b in meta.get("test_batteries", []))
    all_known = train | cal | test
    all_data  = set(df["battery_id"].dropna().astype(str).unique())

    ok = True

    print("=" * 60)
    print("BATTERY SPLIT INTEGRITY CHECK")
    print("=" * 60)

    print(f"\nActual train batteries  ({len(train)}): {sorted(train)}")
    print(f"Calibration batteries   ({len(cal)}):  {sorted(cal)}")
    print(f"Test batteries          ({len(test)}): {sorted(test)}")

    # 1. Train ∩ Test
    overlap_tt = train & test
    print(f"\n[{'PASS' if not overlap_tt else 'FAIL'}] Train ∩ Test overlap: {sorted(overlap_tt) or 'none'}")
    if overlap_tt:
        ok = False

    # 2. Train ∩ Cal
    overlap_tc = train & cal
    print(f"[{'PASS' if not overlap_tc else 'FAIL'}] Train ∩ Cal overlap:  {sorted(overlap_tc) or 'none'}")
    if overlap_tc:
        ok = False

    # 3. Cal ∩ Test
    overlap_ct = cal & test
    print(f"[{'PASS' if not overlap_ct else 'FAIL'}] Cal ∩ Test overlap:   {sorted(overlap_ct) or 'none'}")
    if overlap_ct:
        ok = False

    # 4. Partition check: train ∪ cal ∪ test == all batteries in data
    missing = all_data - all_known
    extra   = all_known - all_data
    print(f"\n[{'PASS' if not missing else 'WARN'}] Batteries in data but not in any split: {sorted(missing) or 'none'}")
    print(f"[{'PASS' if not extra   else 'WARN'}] Batteries in split but not in data:    {sorted(extra) or 'none'}")

    # 5. Dataset distribution
    if "source_dataset" in pd.read_csv(BASE / "data/processed/cycle_features_with_rul.csv", nrows=1).columns:
        full_df = pd.read_csv(BASE / "data/processed/cycle_features_with_rul.csv",
                              usecols=["battery_id", "source_dataset"])
        full_df["battery_id"] = full_df["battery_id"].astype(str)
        for split_name, split_set in [("train", train), ("cal", cal), ("test", test)]:
            sub = full_df[full_df["battery_id"].isin(split_set)]
            dist = sub.drop_duplicates("battery_id")["source_dataset"].value_counts().to_dict()
            print(f"  {split_name:8s} dataset dist: {dist}")

    # 6. Temperature distribution
    temp_map = meta.get("train_temp_mix", {})
    if temp_map:
        print(f"\n  Train temp mix:  {meta.get('train_temp_mix', {})}")
        print(f"  Cal temp mix:    {meta.get('cal_temp_mix', {})}")
        print(f"  Test temp mix:   {meta.get('test_temp_mix', {})}")

    print("\n" + "=" * 60)
    if ok:
        print("RESULT: ALL CHECKS PASSED — no leakage detected")
    else:
        print("RESULT: LEAKAGE DETECTED — fix split before running experiments")
    print("=" * 60)

    return ok


def main() -> None:
    metrics_path = BASE / "trained_models" / "model_metrics.json"
    csv_path     = BASE / "data/processed/cycle_features_with_rul.csv"

    meta = _load_split_metadata(metrics_path)
    df   = _load_features(csv_path)

    ok = check_split(meta, df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

# scripts/run_calce_lobo_cv.py
# Updated: 2026-04-16
"""
Leave-One-Battery-Out Cross-Validation for CALCE CS2 batteries.

Rotates through CS2_35, CS2_36, CS2_37, CS2_38:
  - Train on 3 CALCE + all NASA train batteries
  - Test on the left-out CALCE battery
  - Reports per-rotation RMSE/MAE and mean ± std

Usage:
    python -m scripts.run_calce_lobo_cv
    python scripts/run_calce_lobo_cv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.modeling.ml_model import FEATURE_COLUMNS, XGBoostRULModel

OUTPUT_DIR = BASE / "data/processed/modeling"
DATA_PATH  = BASE / "data/processed/cycle_features_with_rul.csv"

CALCE_BATTERIES = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]


def _is_calce(bid: str) -> bool:
    return str(bid).upper().startswith("CS")


def run_calce_lobo_cv() -> dict:
    if not DATA_PATH.exists():
        print(f"[ERROR] Data not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df["battery_id"] = df["battery_id"].astype(str)

    all_batteries = sorted(df["battery_id"].unique())
    nasa_batteries = [b for b in all_batteries if not _is_calce(b)]
    calce_batteries = [b for b in all_batteries if _is_calce(b)]

    if not calce_batteries:
        print("[ERROR] No CALCE batteries found in dataset.")
        sys.exit(1)

    print("=" * 60)
    print("CALCE LEAVE-ONE-BATTERY-OUT CROSS-VALIDATION")
    print("=" * 60)
    print(f"CALCE batteries: {calce_batteries}")
    print(f"NASA batteries:  {len(nasa_batteries)} total")
    print()

    fold_results = []

    for left_out in calce_batteries:
        calce_train = [b for b in calce_batteries if b != left_out]
        train_batteries = nasa_batteries + calce_train
        test_batteries  = [left_out]

        train_df = df[df["battery_id"].isin(train_batteries)].copy()
        test_df  = df[df["battery_id"].isin(test_batteries)].copy()

        if test_df.empty or "RUL" not in test_df.columns:
            print(f"  [{left_out}] Skipped — no test data or RUL column missing")
            continue

        model = XGBoostRULModel()
        model.fit(train_df, quantiles=False)

        preds  = model.predict(test_df).astype(float)
        actual = test_df["RUL"].values.astype(float)

        valid = np.isfinite(preds) & np.isfinite(actual)
        if valid.sum() == 0:
            print(f"  [{left_out}] Skipped — no valid predictions")
            continue

        rmse = float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))
        mae  = float(np.mean(np.abs(preds[valid] - actual[valid])))

        fold_results.append({
            "left_out_battery": left_out,
            "n_train_batteries": len(train_batteries),
            "n_test_cycles":     int(valid.sum()),
            "rmse":              round(rmse, 4),
            "mae":               round(mae,  4),
        })

        print(f"  Fold [{left_out}]  RMSE={rmse:.2f}  MAE={mae:.2f}  "
              f"(test cycles={valid.sum()}, train batteries={len(train_batteries)})")

    if not fold_results:
        print("[ERROR] No folds completed.")
        sys.exit(1)

    rmse_vals = [f["rmse"] for f in fold_results]
    mae_vals  = [f["mae"]  for f in fold_results]

    summary = {
        "protocol":          "leave_one_calce_battery_out",
        "calce_batteries":   calce_batteries,
        "n_folds":           len(fold_results),
        "mean_rmse":         round(float(np.mean(rmse_vals)), 4),
        "std_rmse":          round(float(np.std(rmse_vals)),  4),
        "mean_mae":          round(float(np.mean(mae_vals)),  4),
        "std_mae":           round(float(np.std(mae_vals)),   4),
        "per_fold":          fold_results,
    }

    print()
    print("=" * 60)
    print(f"CALCE LOBO-CV SUMMARY")
    print(f"  Mean RMSE : {summary['mean_rmse']:.2f} ± {summary['std_rmse']:.2f} cycles")
    print(f"  Mean MAE  : {summary['mean_mae']:.2f} ± {summary['std_mae']:.2f} cycles")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "calce_lobo_cv_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    run_calce_lobo_cv()

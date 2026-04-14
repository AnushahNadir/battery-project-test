from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

logger = logging.getLogger(__name__)

from src.log import setup_logging
from src.modeling.dl_sequence_model import HAS_TORCH, TemporalSequenceModel, train_dl_sequence_model


def _build_search_space() -> List[Dict]:
    # Compact grid to keep runtime reasonable on CPU.
    return [
        {"sequence_length": 8, "hidden_channels": 32, "learning_rate": 1.0e-3, "dropout": 0.10},
        {"sequence_length": 8, "hidden_channels": 48, "learning_rate": 1.5e-3, "dropout": 0.10},
        {"sequence_length": 8, "hidden_channels": 64, "learning_rate": 1.5e-3, "dropout": 0.15},
        {"sequence_length": 12, "hidden_channels": 32, "learning_rate": 1.0e-3, "dropout": 0.10},
        {"sequence_length": 12, "hidden_channels": 48, "learning_rate": 1.5e-3, "dropout": 0.10},
        {"sequence_length": 12, "hidden_channels": 64, "learning_rate": 2.0e-3, "dropout": 0.15},
        {"sequence_length": 16, "hidden_channels": 32, "learning_rate": 1.0e-3, "dropout": 0.10},
        {"sequence_length": 16, "hidden_channels": 48, "learning_rate": 1.5e-3, "dropout": 0.10},
        {"sequence_length": 16, "hidden_channels": 64, "learning_rate": 2.0e-3, "dropout": 0.15},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune sequence model hyperparameters.")
    parser.add_argument("--input", default="data/processed/cycle_features_with_rul.csv")
    parser.add_argument("--model-dir", default="trained_models")
    parser.add_argument("--out-dir", default="data/processed/modeling")
    parser.add_argument("--max-trials", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--select-by",
        choices=["val_rmse", "test_probe_rmse"],
        default="val_rmse",
        help="How to pick the best config from trials.",
    )
    args = parser.parse_args()
    setup_logging()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    batteries = sorted(df["battery_id"].dropna().astype(str).unique())
    n_total = len(batteries)
    n_train = max(1, min(int(0.8 * n_total), n_total - 1))

    train_batteries = batteries[:n_train]
    test_batteries = batteries[n_train:]

    n_inner_train = max(1, min(int(0.8 * len(train_batteries)), len(train_batteries) - 1))
    inner_train_bats = train_batteries[:n_inner_train]
    inner_val_bats = train_batteries[n_inner_train:]

    inner_train_df = df[df["battery_id"].astype(str).isin(inner_train_bats)].copy()
    inner_val_df = df[df["battery_id"].astype(str).isin(inner_val_bats)].copy()
    test_df = df[df["battery_id"].astype(str).isin(test_batteries)].copy()

    force_backend = "tcn" if HAS_TORCH else "mlp"
    logger.info(f"Backend for tuning: {force_backend}")
    logger.info(
        f"Battery split: total={n_total}, train={len(train_batteries)}, "
        f"test={len(test_batteries)}, tune_train={len(inner_train_bats)}, tune_val={len(inner_val_bats)}"
    )

    search_space = _build_search_space()[: args.max_trials]
    results = []

    for i, cfg in enumerate(search_space, start=1):
        logger.info(f"\nTrial {i}/{len(search_space)}: {cfg}")
        model = TemporalSequenceModel(
            sequence_length=cfg["sequence_length"],
            force_backend=force_backend,
            hidden_channels=cfg["hidden_channels"],
            learning_rate=cfg["learning_rate"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=12,
            dropout=cfg["dropout"],
            weight_decay=1e-5,
        )
        model.fit(inner_train_df)
        val_metrics = model.evaluate(inner_val_df)
        test_metrics = model.evaluate(test_df)

        row = {
            "trial": i,
            "config": cfg,
            "val_metrics": val_metrics.to_dict(),
            "test_metrics_probe": test_metrics.to_dict(),
        }
        results.append(row)
        logger.info(
            f"  val_rmse={val_metrics.rmse:.3f} "
            f"test_probe_rmse={test_metrics.rmse:.3f} backend={val_metrics.backend}"
        )

    if args.select_by == "val_rmse":
        best = min(results, key=lambda r: r["val_metrics"]["rmse"])
    else:
        best = min(results, key=lambda r: r["test_metrics_probe"]["rmse"])
    best_cfg = dict(best["config"])

    logger.info(f"\nBest config by {args.select_by}:")
    logger.info(best_cfg)

    final_model, final_metrics = train_dl_sequence_model(
        df=df,
        train_batteries=train_batteries,
        test_batteries=test_batteries,
        output_dir=model_dir,
        sequence_length=best_cfg["sequence_length"],
        hidden_channels=best_cfg["hidden_channels"],
        learning_rate=best_cfg["learning_rate"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=12,
        dropout=best_cfg["dropout"],
        weight_decay=1e-5,
        force_backend=force_backend,
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "backend": force_backend,
        "data_split": {
            "total_batteries": n_total,
            "train_batteries": len(train_batteries),
            "test_batteries": len(test_batteries),
            "inner_train_batteries": len(inner_train_bats),
            "inner_val_batteries": len(inner_val_bats),
        },
        "search_space_size": len(search_space),
        "best_config": best_cfg,
        "selection_metric": args.select_by,
        "best_val_metrics": best["val_metrics"],
        "best_test_probe_metrics": best["test_metrics_probe"],
        "final_test_metrics": final_metrics.to_dict(),
        "trials": results,
    }

    out_json = out_dir / "sequence_tuning_results.json"
    out_md = out_dir / "sequence_tuning_report.md"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Sequence Model Tuning Report",
        "",
        f"Generated: {summary['created_at']}",
        f"Backend: `{force_backend}`",
        "",
        f"## Best Config (by {args.select_by})",
        "",
        f"- sequence_length: {best_cfg['sequence_length']}",
        f"- hidden_channels: {best_cfg['hidden_channels']}",
        f"- learning_rate: {best_cfg['learning_rate']}",
        f"- dropout: {best_cfg['dropout']}",
        "",
        "## Performance",
        "",
        f"- Best validation RMSE: {best['val_metrics']['rmse']:.3f}",
        f"- Best test-probe RMSE: {best['test_metrics_probe']['rmse']:.3f}",
        f"- Final test RMSE: {final_metrics.rmse:.3f}",
        f"- Final test MAE: {final_metrics.mae:.3f}",
        f"- Final backend: {final_metrics.backend}",
        "",
        "## Notes",
        "",
        "- Selection used battery-level inner train/validation split.",
        "- Final model was retrained on full train batteries and evaluated on held-out test batteries.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"\nSaved tuning results: {out_json}")
    logger.info(f"Saved tuning report: {out_md}")


if __name__ == "__main__":
    main()

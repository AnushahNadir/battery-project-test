# src/modeling/anomaly_detection.py
"""
Stage 4.5: Anomaly Detection

Purpose
-------
Detect abnormal degradation behavior to SUPPORT human analysis.

Definition
----------
An anomaly is a cycle whose degradation behavior deviates strongly
from the learned population trend, measured via model residuals.

Guarantees
----------
- No causal claims
- No automatic decisions
- Model-grounded only
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------

@dataclass
class Anomaly:
    battery_id: str
    cycle_index: int
    anomaly_score: float
    anomaly_type: str
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "battery_id": self.battery_id,
            "cycle_index": int(self.cycle_index),
            "anomaly_score": round(float(self.anomaly_score), 3),
            "anomaly_type": self.anomaly_type,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    model,
    top_fraction: float = 0.05,
) -> List[Anomaly]:
    """
    Detect anomalies using model residuals.

    Parameters
    ----------
    df : DataFrame
        cycle_features_with_rul table
    model :
        trained ML model with predict()
    top_fraction : float
        fraction of most abnormal points to flag (default 5%)

    Returns
    -------
    List[Anomaly]
    """

    # Predict RUL
    preds = model.predict(df)
    actuals = df["RUL"].values

    residuals = actuals - preds
    abs_residuals = np.abs(residuals)

    # Threshold based on percentile
    threshold = np.quantile(abs_residuals, 1 - top_fraction)

    anomalies: List[Anomaly] = []

    for pos, (_, row) in enumerate(df.iterrows()):
        score = abs_residuals[pos]
        if score >= threshold:
            anomalies.append(
                Anomaly(
                    battery_id=str(row["battery_id"]),
                    cycle_index=int(row["cycle_index"]),
                    anomaly_score=float(score),
                    anomaly_type="residual_spike",
                    explanation=(
                        "Observed RUL deviates strongly from model-predicted trend. "
                        "This may indicate abnormal degradation behavior."
                    ),
                )
            )

    return anomalies


# ---------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------

def run_anomaly_detection(
    df: pd.DataFrame,
    ml_model,
    output_dir: Path,
) -> List[Anomaly]:
    """
    Run anomaly detection and write anomalies.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running anomaly detection...")

    anomalies = detect_anomalies(df, ml_model)

    out_path = output_dir / "anomalies.json"
    out_path.write_text(
        json.dumps([a.to_dict() for a in anomalies], indent=2),
        encoding="utf-8",
    )

    logger.info(f"  Detected {len(anomalies)} anomalies")
    logger.info(f"  Saved to {out_path}")

    return anomalies

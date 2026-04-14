# src/modeling/statistical_baseline.py
"""
Statistical Baseline Model: Exponential Capacity Fade

Model: Capacity(t) = C0 * exp(-λ * t)
Where:
    C0 = initial capacity
    λ = degradation rate (fitted per battery)
    t = cycle index

This provides a physics-informed baseline for battery degradation.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass
class ExponentialFitResult:
    """Result of exponential fit for a single battery."""
    battery_id: str
    c0: float  # Initial capacity
    lambda_: float  # Degradation rate
    r_squared: float
    rmse: float
    n_cycles: int
    
    def to_dict(self) -> Dict:
        return {
            "battery_id": self.battery_id,
            "c0": self.c0,
            "lambda": self.lambda_,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "n_cycles": self.n_cycles,
        }


@dataclass
class StatisticalModelMetrics:
    """Metrics for statistical baseline model."""
    rmse: float
    mae: float
    trajectory_shape_error: float
    battery_generalization_score: float
    per_battery_rmse: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "trajectory_shape_error": self.trajectory_shape_error,
            "battery_generalization_score": self.battery_generalization_score,
            "per_battery_rmse": self.per_battery_rmse,
        }


class StatisticalBaselineModel:
    """
    Exponential capacity fade model for battery degradation.
    
    Fits Capacity(t) = C0 * exp(-λ * t) per battery, then uses
    the degradation rate to estimate RUL based on EOL threshold.
    """
    
    def __init__(self):
        self.battery_params: Dict[str, ExponentialFitResult] = {}
        self.global_lambda_mean: float = 0.0
        self.global_lambda_std: float = 0.0
        self.residuals: Dict[str, np.ndarray] = {}

    @staticmethod
    def _sanitize_fit_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Return rows safe for exponential fitting."""
        work = df.copy()
        work["cycle_index"] = pd.to_numeric(work["cycle_index"], errors="coerce")
        work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce")
        work["init_capacity"] = pd.to_numeric(work["init_capacity"], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["cycle_index", "capacity", "init_capacity"]).copy()
        work = work[(work["capacity"] > 0) & (work["init_capacity"] > 0)].copy()
        if work.empty:
            return work
        work["cycle_index"] = work["cycle_index"].astype(int)
        return work.sort_values("cycle_index")

    @staticmethod
    def _sanitize_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Return rows safe for RUL evaluation."""
        work = df.copy()
        work["cycle_index"] = pd.to_numeric(work["cycle_index"], errors="coerce")
        work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce")
        work["eol_capacity_threshold"] = pd.to_numeric(work["eol_capacity_threshold"], errors="coerce")
        work["RUL"] = pd.to_numeric(work["RUL"], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["battery_id", "cycle_index", "capacity", "eol_capacity_threshold", "RUL"]).copy()
        work = work[(work["capacity"] > 0) & (work["eol_capacity_threshold"] > 0)].copy()
        if work.empty:
            return work
        work["cycle_index"] = work["cycle_index"].astype(int)
        work["battery_id"] = work["battery_id"].astype(str)
        return work
        
    @staticmethod
    def exponential_decay(t: np.ndarray, c0: float, lambda_: float) -> np.ndarray:
        """Exponential decay function."""
        return c0 * np.exp(-lambda_ * t)
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit exponential model for each battery.
        
        Args:
            df: DataFrame with columns ['battery_id', 'cycle_index', 'capacity', 'init_capacity']
        """
        self.battery_params = {}
        self.residuals = {}
        
        lambdas = []
        
        for battery_id in df['battery_id'].unique():
            batt_df = self._sanitize_fit_rows(df[df['battery_id'] == battery_id])
            if len(batt_df) < 3:
                if len(batt_df) == 0:
                    logger.warning(f"Battery {battery_id} has no valid rows after numeric sanitation.")
                else:
                    logger.warning(f"Battery {battery_id} has only {len(batt_df)} valid rows; using defaults.")
                default_c0 = 1.0
                if "init_capacity" in batt_df.columns and len(batt_df):
                    default_c0 = float(batt_df["init_capacity"].iloc[0])
                self.battery_params[battery_id] = ExponentialFitResult(
                    battery_id=battery_id,
                    c0=max(default_c0, 1e-6),
                    lambda_=0.002,
                    r_squared=0.0,
                    rmse=0.0,
                    n_cycles=len(batt_df),
                )
                self.residuals[battery_id] = np.array([], dtype=float)
                continue

            t = batt_df['cycle_index'].values.astype(float)
            capacity = batt_df['capacity'].values.astype(float)
            c0_init = float(batt_df['init_capacity'].iloc[0])
            
            try:
                # Fit exponential decay
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        self.exponential_decay,
                        t, capacity,
                        p0=[c0_init, 0.001],
                        bounds=([0.5 * c0_init, 0.0], [2.0 * c0_init, 0.1]),
                        maxfev=5000,
                    )
                
                c0_fit, lambda_fit = popt
                
                # Calculate metrics
                pred = self.exponential_decay(t, c0_fit, lambda_fit)
                residuals = capacity - pred
                rmse = np.sqrt(np.mean(residuals ** 2))
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((capacity - np.mean(capacity)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                self.battery_params[battery_id] = ExponentialFitResult(
                    battery_id=battery_id,
                    c0=c0_fit,
                    lambda_=lambda_fit,
                    r_squared=r_squared,
                    rmse=rmse,
                    n_cycles=len(t),
                )
                
                self.residuals[battery_id] = residuals
                lambdas.append(lambda_fit)
                
            except Exception as e:
                logger.warning(f"Failed to fit battery {battery_id}: {e}")
                # Use default parameters
                fallback_c0 = float(np.nanmedian(capacity)) if len(capacity) else c0_init
                if not np.isfinite(fallback_c0) or fallback_c0 <= 0:
                    fallback_c0 = max(c0_init, 1e-6)
                self.battery_params[battery_id] = ExponentialFitResult(
                    battery_id=battery_id,
                    c0=fallback_c0,
                    lambda_=0.002,  # Default degradation rate
                    r_squared=0.0,
                    rmse=float(np.sqrt(np.mean((capacity - fallback_c0) ** 2))) if len(capacity) else 0.0,
                    n_cycles=len(t),
                )
                self.residuals[battery_id] = np.array([], dtype=float)
        
        # Compute global statistics
        if lambdas:
            self.global_lambda_mean = np.mean(lambdas)
            self.global_lambda_std = np.std(lambdas)
        else:
            self.global_lambda_mean = 0.002
            self.global_lambda_std = 0.0
    
    def predict_rul(self, battery_id: str, current_cycle: int, 
                    current_capacity: float, eol_threshold: float) -> float:
        """
        Predict remaining useful life based on exponential model.
        
        Args:
            battery_id: Battery identifier
            current_cycle: Current cycle number
            current_capacity: Current capacity
            eol_threshold: End-of-life capacity threshold
            
        Returns:
            Predicted RUL in cycles
        """
        if battery_id in self.battery_params:
            params = self.battery_params[battery_id]
            lambda_ = params.lambda_
            c0 = params.c0
        else:
            # Use global average for unseen batteries
            lambda_ = self.global_lambda_mean if self.global_lambda_mean > 0 else 0.002
            c0 = current_capacity * np.exp(lambda_ * current_cycle)
        
        if lambda_ <= 0 or current_capacity <= eol_threshold:
            return 0.0
        
        # Predict EOL cycle: C0 * exp(-λ * t_eol) = threshold
        # t_eol = -ln(threshold / C0) / λ
        try:
            t_eol = -np.log(eol_threshold / c0) / lambda_
            rul = max(0, t_eol - current_cycle)
            return rul
        except:
            return 0.0
    
    def predict_rul_with_uncertainty(self, battery_id: str, current_cycle: int,
                                     current_capacity: float, eol_threshold: float,
                                     n_bootstrap: int = 100) -> Tuple[float, float, float]:
        """
        Predict RUL with bootstrap uncertainty intervals.
        
        Returns:
            (rul_lower_5, rul_median, rul_upper_95)
        """
        rul_samples = []
        
        base_rul = self.predict_rul(battery_id, current_cycle, 
                                     current_capacity, eol_threshold)
        
        # Get residuals for bootstrap
        if battery_id in self.residuals:
            residuals = self.residuals[battery_id]
        else:
            residuals = np.concatenate(list(self.residuals.values()))
        
        residual_std = np.std(residuals) if len(residuals) > 0 else 0.05
        
        # Bootstrap predictions
        for _ in range(n_bootstrap):
            # Perturb capacity by residual
            perturbed_capacity = current_capacity + np.random.normal(0, residual_std)
            perturbed_rul = self.predict_rul(battery_id, current_cycle,
                                             perturbed_capacity, eol_threshold)
            rul_samples.append(perturbed_rul)
        
        rul_samples = np.array(rul_samples)
        
        return (
            float(np.percentile(rul_samples, 5)),
            float(np.percentile(rul_samples, 50)),
            float(np.percentile(rul_samples, 95)),
        )
    
    def predict_rul_batch_with_uncertainty(
        self,
        battery_ids: "np.ndarray",
        cycle_indices: "np.ndarray",
        capacities: "np.ndarray",
        eol_thresholds: "np.ndarray",
        n_bootstrap: int = 100,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Vectorized batch version of predict_rul_with_uncertainty.

        Returns:
            (lower_5, median_50, upper_95) each as a float64 array of shape (n,).
        """
        battery_ids = np.asarray(battery_ids, dtype=object)
        cycle_indices = np.asarray(cycle_indices, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        eol_thresholds = np.asarray(eol_thresholds, dtype=float)
        n = len(battery_ids)

        lambdas = np.empty(n, dtype=float)
        c0s = np.empty(n, dtype=float)
        residual_stds = np.empty(n, dtype=float)

        # Pooled std for batteries not in training data.
        if self.residuals:
            pooled = np.concatenate(list(self.residuals.values()))
            pooled_std = float(np.std(pooled)) if len(pooled) > 0 else 0.05
        else:
            pooled_std = 0.05

        for i, bid in enumerate(battery_ids):
            bid = str(bid)
            if bid in self.battery_params:
                params = self.battery_params[bid]
                lambdas[i] = params.lambda_
                c0s[i] = params.c0
            else:
                lam = self.global_lambda_mean if self.global_lambda_mean > 0 else 0.002
                lambdas[i] = lam
                c0s[i] = capacities[i] * np.exp(lam * cycle_indices[i])

            if bid in self.residuals:
                res = self.residuals[bid]
                residual_stds[i] = float(np.std(res)) if len(res) > 0 else 0.05
            else:
                residual_stds[i] = pooled_std

        known_mask = np.array([str(bid) in self.battery_params for bid in battery_ids])

        # Bootstrap noise: (n, n_bootstrap)
        noise = np.random.standard_normal((n, n_bootstrap)) * residual_stds[:, None]
        perturbed_caps = capacities[:, None] + noise

        lam_col = lambdas[:, None]
        cyc_col = cycle_indices[:, None]
        eol_col = eol_thresholds[:, None]

        # For unknown batteries the initial capacity c0 depends on perturbed capacity.
        c0_boot = np.where(
            known_mask[:, None],
            c0s[:, None],
            perturbed_caps * np.exp(lam_col * cyc_col),
        )

        valid = (lam_col > 0) & (capacities[:, None] > eol_col)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_eol = np.where(valid, -np.log(eol_col / c0_boot) / lam_col, 0.0)

        rul_boot = np.maximum(0.0, t_eol - cyc_col)

        lower = np.percentile(rul_boot, 5, axis=1)
        median = np.percentile(rul_boot, 50, axis=1)
        upper = np.percentile(rul_boot, 95, axis=1)

        return lower, median, upper

    def evaluate(self, df: pd.DataFrame) -> StatisticalModelMetrics:
        """
        Evaluate model on test data.
        
        Args:
            df: DataFrame with RUL ground truth
            
        Returns:
            Model metrics
        """
        predictions = []
        actuals = []
        per_battery_errors = {}
        eval_df = self._sanitize_eval_rows(df)
        if eval_df.empty:
            return StatisticalModelMetrics(
                rmse=float("inf"),
                mae=float("inf"),
                trajectory_shape_error=1.0,
                battery_generalization_score=float("inf"),
                per_battery_rmse={},
            )
        
        for battery_id in eval_df['battery_id'].unique():
            batt_df = eval_df[eval_df['battery_id'] == battery_id]
            batt_errors = []
            
            for _, row in batt_df.iterrows():
                rul_pred = self.predict_rul(
                    battery_id,
                    int(row['cycle_index']),
                    float(row['capacity']),
                    float(row['eol_capacity_threshold']),
                )
                rul_actual = float(row['RUL'])
                
                predictions.append(rul_pred)
                actuals.append(rul_actual)
                batt_errors.append((rul_pred - rul_actual) ** 2)
            if batt_errors:
                per_battery_errors[battery_id] = np.sqrt(np.mean(batt_errors))
        
        if not predictions:
            return StatisticalModelMetrics(
                rmse=float("inf"),
                mae=float("inf"),
                trajectory_shape_error=1.0,
                battery_generalization_score=float("inf"),
                per_battery_rmse={},
            )
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        mae = float(np.mean(np.abs(predictions - actuals)))
        
        # Trajectory shape error (simplified DTW approximation)
        # Using correlation-based shape similarity
        trajectory_corr = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else np.nan
        trajectory_shape_error = float(1 - abs(trajectory_corr)) if not np.isnan(trajectory_corr) else 1.0
        
        # Battery generalization score (mean of per-battery RMSE)
        battery_gen_score = float(np.mean(list(per_battery_errors.values()))) if per_battery_errors else float("inf")
        
        return StatisticalModelMetrics(
            rmse=rmse,
            mae=mae,
            trajectory_shape_error=trajectory_shape_error,
            battery_generalization_score=battery_gen_score,
            per_battery_rmse={k: float(v) for k, v in per_battery_errors.items()},
        )
    
    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'battery_params': {k: v.to_dict() for k, v in self.battery_params.items()},
                'global_lambda_mean': self.global_lambda_mean,
                'global_lambda_std': self.global_lambda_std,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'StatisticalBaselineModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.global_lambda_mean = data['global_lambda_mean']
        model.global_lambda_std = data['global_lambda_std']
        
        for k, v in data['battery_params'].items():
            model.battery_params[k] = ExponentialFitResult(
                battery_id=v['battery_id'],
                c0=v['c0'],
                lambda_=v['lambda'],
                r_squared=v['r_squared'],
                rmse=v['rmse'],
                n_cycles=v['n_cycles'],
            )
        
        return model


def train_statistical_baseline(
    df: pd.DataFrame,
    train_batteries: List[str],
    test_batteries: List[str],
    output_dir: Path,
) -> Tuple[StatisticalBaselineModel, StatisticalModelMetrics]:
    """
    Train and evaluate statistical baseline model.
    
    Args:
        df: Full dataset
        train_batteries: Battery IDs for training
        test_batteries: Battery IDs for testing
        output_dir: Directory for saving model
        
    Returns:
        Trained model and metrics
    """
    # Split data by battery (NO RANDOM LEAKAGE)
    train_df = df[df['battery_id'].isin(train_batteries)].copy()
    test_df = df[df['battery_id'].isin(test_batteries)].copy()
    
    logger.info(f"Training statistical baseline on {len(train_batteries)} batteries...")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    # Train model
    model = StatisticalBaselineModel()
    model.fit(train_df)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(test_df)

    logger.info(f"  RMSE: {metrics.rmse:.2f} cycles")
    logger.info(f"  MAE: {metrics.mae:.2f} cycles")
    logger.info(f"  Battery Generalization: {metrics.battery_generalization_score:.2f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "statistical_baseline.pkl"
    model.save(model_path)
    logger.info(f"  Saved model to {model_path}")
    
    return model, metrics


if __name__ == "__main__":
    # Test the model
    base = Path("d:/Energy Project/code/battery-project3")
    df = pd.read_csv(base / "data/processed/cycle_features_with_rul.csv")
    
    batteries = df['battery_id'].unique().tolist()
    n_train = int(len(batteries) * 0.8)
    train_batteries = batteries[:n_train]
    test_batteries = batteries[n_train:]
    
    model, metrics = train_statistical_baseline(
        df, train_batteries, test_batteries,
        base / "trained_models"
    )
    
    logger.info("\nMetrics:")
    logger.info(json.dumps(metrics.to_dict(), indent=2))

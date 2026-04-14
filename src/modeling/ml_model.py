# src/modeling/ml_model.py
"""
Machine Learning Model: XGBoost for RUL Prediction

Features used:
- capacity: Current degradation state
- temp_mean, temp_max: Thermal stress indicators
- v_min, v_mean: Voltage profile
- i_mean, i_min: Current profile
- energy_j, ah_est: Energy metrics
- duration_s: Cycle duration

Target: RUL (Remaining Useful Life)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class MLModelMetrics:
    """Metrics for ML model."""
    rmse: float
    mae: float
    trajectory_shape_error: float
    battery_generalization_score: float
    per_battery_rmse: Dict[str, float]
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "trajectory_shape_error": self.trajectory_shape_error,
            "battery_generalization_score": self.battery_generalization_score,
            "per_battery_rmse": self.per_battery_rmse,
            "feature_importance": self.feature_importance,
        }


@dataclass
class GroupKFoldFoldMetrics:
    """Per-fold metrics for GroupKFold cross-validation."""
    fold_index: int
    n_train_rows: int
    n_val_rows: int
    n_train_batteries: int
    n_val_batteries: int
    rmse: float
    mae: float

    def to_dict(self) -> Dict:
        return {
            "fold_index": self.fold_index,
            "n_train_rows": self.n_train_rows,
            "n_val_rows": self.n_val_rows,
            "n_train_batteries": self.n_train_batteries,
            "n_val_batteries": self.n_val_batteries,
            "rmse": self.rmse,
            "mae": self.mae,
        }


@dataclass
class GroupKFoldCVMetrics:
    """Summary metrics for GroupKFold battery-level cross-validation."""
    n_splits: int
    n_batteries: int
    mean_rmse: float
    std_rmse: float
    mean_mae: float
    std_mae: float
    folds: List[GroupKFoldFoldMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "strategy": "GroupKFold(groups=battery_id)",
            "group_column": "battery_id",
            "n_splits": self.n_splits,
            "n_batteries": self.n_batteries,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_mae": self.mean_mae,
            "std_mae": self.std_mae,
            "folds": [f.to_dict() for f in self.folds],
        }


# Feature columns for modeling
FEATURE_COLUMNS = [
    'capacity',
    'cycle_index',
    'temp_mean',
    'temp_max',
    'v_mean',
    'i_mean',
    'i_min',
    'energy_j',
    'ah_est',
    'duration_s',
]


class XGBoostRULModel:
    """
    XGBoost model for RUL prediction with uncertainty quantification.
    
    Uses quantile regression for uncertainty estimation.
    """
    
    def __init__(self, feature_columns: List[str] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.model = None
        self.model_lower = None  # 5th percentile
        self.model_upper = None  # 95th percentile
        self.feature_importance_: Dict[str, float] = {}
        self._use_ensemble_variance = False
        
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare a numeric feature matrix from dataframe."""
        X = df.reindex(columns=self.feature_columns).copy()

        for col in self.feature_columns:
            s = X[col]
            if s.dtype == object:
                s = s.astype(str).str.strip().str.replace(r"^\[|\]$", "", regex=True)
            X[col] = pd.to_numeric(s, errors="coerce")

        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        return X.values
    
    def fit(self, df: pd.DataFrame, quantiles: bool = True) -> None:
        """
        Fit XGBoost models for median and quantiles.
        
        Args:
            df: Training data with RUL column
            quantiles: Whether to train quantile models for uncertainty
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed, using Random Forest fallback")
            self._fit_rf_fallback(df, quantiles)
            return
        
        X = self._prepare_features(df)
        y = df['RUL'].values
        
        cfg = get_config()
        xcfg = cfg.xgboost

        # Main model (median/mean prediction)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': xcfg.max_depth,
            'learning_rate': xcfg.learning_rate,
            'n_estimators': xcfg.n_estimators,
            'subsample': xcfg.subsample,
            'colsample_bytree': xcfg.colsample_bytree,
            'random_state': xcfg.random_state,
        }
        
        try:
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X, y)
            
            # Store feature importance
            importance = self.model.feature_importances_
            self.feature_importance_ = {
                col: float(imp) for col, imp in zip(self.feature_columns, importance)
            }
            
            if quantiles:
                low_alpha = min([q for q in xcfg.quantile_alphas if q < 0.5], default=0.05)
                high_alpha = max([q for q in xcfg.quantile_alphas if q > 0.5], default=0.95)
                # Lower bound (5th percentile)
                params_lower = params.copy()
                params_lower['objective'] = 'reg:quantileerror'
                params_lower['quantile_alpha'] = low_alpha
                self.model_lower = xgb.XGBRegressor(**params_lower)
                self.model_lower.fit(X, y)
                
                # Upper bound (95th percentile)
                params_upper = params.copy()
                params_upper['objective'] = 'reg:quantileerror'
                params_upper['quantile_alpha'] = high_alpha
                self.model_upper = xgb.XGBRegressor(**params_upper)
                self.model_upper.fit(X, y)

        except Exception as e:
            logger.warning(f"XGBoost fit failed ({e}), falling back to Random Forest")
            self._fit_rf_fallback(df, quantiles)
    
    def _fit_rf_fallback(self, df: pd.DataFrame, quantiles: bool = True) -> None:
        """Fallback to Random Forest if XGBoost not available."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = self._prepare_features(df)
        y = df['RUL'].values
        cfg = get_config()
        xcfg = cfg.xgboost
        
        self.model = RandomForestRegressor(
            n_estimators=xcfg.n_estimators,
            max_depth=xcfg.max_depth,
            random_state=xcfg.random_state,
        )
        self.model.fit(X, y)
        
        # Store feature importance
        importance = self.model.feature_importances_
        self.feature_importance_ = {
            col: float(imp) for col, imp in zip(self.feature_columns, importance)
        }
        
        # For uncertainty, we'll use ensemble variance
        self.model_lower = None
        self.model_upper = None
        self._use_ensemble_variance = True
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict RUL for given data."""
        X = self._prepare_features(df)
        return self.model.predict(X)

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict raw quantiles (unclipped) for RUL.

        Returns DataFrame with columns: q05, q50, q95.
        """
        X = self._prepare_features(df)
        q50 = self.model.predict(X).astype(float)

        if self.model_lower is not None and self.model_upper is not None:
            q05 = self.model_lower.predict(X).astype(float)
            q95 = self.model_upper.predict(X).astype(float)
        elif self._use_ensemble_variance:
            # Use RF ensemble variance as quantiles.
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_], dtype=float)
            q05 = np.percentile(predictions, 5, axis=0).astype(float)
            q95 = np.percentile(predictions, 95, axis=0).astype(float)
        else:
            # Conservative symmetric fallback around median.
            spread = np.maximum(np.abs(q50) * 0.3, 1.0)
            q05 = (q50 - spread).astype(float)
            q95 = (q50 + spread).astype(float)

        # Ensure monotonic ordering.
        lo = np.minimum(q05, q95)
        hi = np.maximum(q05, q95)
        return pd.DataFrame({"q05": lo, "q50": q50, "q95": hi})
    
    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict RUL with uncertainty bounds.
        
        Returns:
            (lower_5, median, upper_95)
        """
        q = self.predict_quantiles(df)
        median = q["q50"].values.astype(float)
        lower = np.maximum(q["q05"].values.astype(float), 0.0)
        upper = np.maximum(q["q95"].values.astype(float), median)
        return lower, median, upper
    
    def evaluate(self, df: pd.DataFrame) -> MLModelMetrics:
        """Evaluate model on test data."""
        predictions = self.predict(df)
        actuals = df['RUL'].values
        
        # Per-battery RMSE
        per_battery_errors = {}
        for battery_id in df['battery_id'].unique():
            batt_mask = df['battery_id'] == battery_id
            batt_pred = predictions[batt_mask]
            batt_actual = actuals[batt_mask]
            per_battery_errors[battery_id] = float(np.sqrt(np.mean((batt_pred - batt_actual) ** 2)))
        
        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        mae = float(np.mean(np.abs(predictions - actuals)))
        
        # Trajectory shape correlation
        trajectory_corr = np.corrcoef(predictions, actuals)[0, 1]
        trajectory_shape_error = float(1 - trajectory_corr) if not np.isnan(trajectory_corr) else 1.0
        
        battery_gen_score = float(np.std(list(per_battery_errors.values())))
        
        return MLModelMetrics(
            rmse=rmse,
            mae=mae,
            trajectory_shape_error=trajectory_shape_error,
            battery_generalization_score=battery_gen_score,
            per_battery_rmse=per_battery_errors,
            feature_importance=self.feature_importance_,
        )
    
    def save(self, path: Path) -> None:
        """Save model to a directory using XGBoost's native format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path / "model.ubj")
        if self.model_lower is not None:
            self.model_lower.save_model(path / "model_lower.ubj")
        if self.model_upper is not None:
            self.model_upper.save_model(path / "model_upper.ubj")
        metadata = {
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance_,
            'has_lower': self.model_lower is not None,
            'has_upper': self.model_upper is not None,
            '_use_ensemble_variance': self._use_ensemble_variance,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> 'XGBoostRULModel':
        """Load model from a directory saved with save()."""
        import xgboost as xgb
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        model = cls(feature_columns=metadata['feature_columns'])
        model.feature_importance_ = metadata['feature_importance']
        model._use_ensemble_variance = metadata['_use_ensemble_variance']
        model.model = xgb.XGBRegressor()
        model.model.load_model(path / "model.ubj")
        if metadata['has_lower']:
            model.model_lower = xgb.XGBRegressor()
            model.model_lower.load_model(path / "model_lower.ubj")
        if metadata['has_upper']:
            model.model_upper = xgb.XGBRegressor()
            model.model_upper.load_model(path / "model_upper.ubj")
        return model


def train_ml_model(
    df: pd.DataFrame,
    train_batteries: List[str],
    test_batteries: List[str],
    output_dir: Path,
) -> Tuple[XGBoostRULModel, MLModelMetrics]:
    """
    Train and evaluate ML model.
    
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
    
    logger.info(f"Training XGBoost model on {len(train_batteries)} batteries...")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    # Train model
    model = XGBoostRULModel()
    model.fit(train_df, quantiles=True)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(test_df)
    
    logger.info(f"  RMSE: {metrics.rmse:.2f} cycles")
    logger.info(f"  MAE: {metrics.mae:.2f} cycles")
    logger.info(f"  Battery Generalization: {metrics.battery_generalization_score:.2f}")
    
    # Show top features
    logger.info("  Top 5 features by importance:")
    sorted_features = sorted(metrics.feature_importance.items(),
                            key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        logger.info(f"    {feat}: {imp:.3f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model"
    model.save(model_path)
    logger.info(f"  Saved model to {model_path}")
    
    return model, metrics


def evaluate_groupkfold_cv(
    df: pd.DataFrame,
    battery_ids: List[str],
    n_splits: int = 5,
) -> GroupKFoldCVMetrics:
    """
    Battery-level cross-validation using GroupKFold(groups=battery_id).

    This prevents any cycle-level leakage across train/validation folds.
    """
    work = df[df["battery_id"].astype(str).isin([str(b) for b in battery_ids])].copy()
    if work.empty:
        raise ValueError("GroupKFold CV cannot run: empty dataset after battery filtering.")
    if "RUL" not in work.columns:
        raise ValueError("GroupKFold CV cannot run: missing required column 'RUL'.")
    if "battery_id" not in work.columns:
        raise ValueError("GroupKFold CV cannot run: missing required column 'battery_id'.")

    groups = work["battery_id"].astype(str).values
    unique_batteries = sorted(set(groups.tolist()))
    n_groups = len(unique_batteries)
    if n_groups < 2:
        raise ValueError("GroupKFold CV requires at least 2 distinct batteries.")

    n_splits_eff = max(2, min(int(n_splits), n_groups))
    splitter = GroupKFold(n_splits=n_splits_eff)

    fold_metrics: List[GroupKFoldFoldMetrics] = []
    logger.info(f"[GroupKFold CV] n_splits={n_splits_eff}, batteries={n_groups}")
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(work, work["RUL"].values, groups), start=1):
        train_df = work.iloc[train_idx].copy()
        val_df = work.iloc[val_idx].copy()

        train_bats = set(train_df["battery_id"].astype(str).unique().tolist())
        val_bats = set(val_df["battery_id"].astype(str).unique().tolist())
        overlap = train_bats & val_bats
        if overlap:
            raise RuntimeError(
                f"GroupKFold leakage detected in fold {fold_idx}: {sorted(overlap)}"
            )

        fold_model = XGBoostRULModel()
        fold_model.fit(train_df, quantiles=False)
        preds = fold_model.predict(val_df).astype(float)
        actual = val_df["RUL"].values.astype(float)
        rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
        mae = float(np.mean(np.abs(preds - actual)))

        fold_metrics.append(
            GroupKFoldFoldMetrics(
                fold_index=fold_idx,
                n_train_rows=int(len(train_df)),
                n_val_rows=int(len(val_df)),
                n_train_batteries=int(len(train_bats)),
                n_val_batteries=int(len(val_bats)),
                rmse=rmse,
                mae=mae,
            )
        )
        logger.info(
            f"  fold={fold_idx}/{n_splits_eff} "
            f"train_batteries={len(train_bats)} val_batteries={len(val_bats)} "
            f"rmse={rmse:.2f} mae={mae:.2f}"
        )

    rmse_vals = np.asarray([f.rmse for f in fold_metrics], dtype=float)
    mae_vals = np.asarray([f.mae for f in fold_metrics], dtype=float)
    summary = GroupKFoldCVMetrics(
        n_splits=n_splits_eff,
        n_batteries=n_groups,
        mean_rmse=float(np.mean(rmse_vals)),
        std_rmse=float(np.std(rmse_vals)),
        mean_mae=float(np.mean(mae_vals)),
        std_mae=float(np.std(mae_vals)),
        folds=fold_metrics,
    )
    logger.info(
        f"  summary: mean_rmse={summary.mean_rmse:.2f} (+/- {summary.std_rmse:.2f}), "
        f"mean_mae={summary.mean_mae:.2f} (+/- {summary.std_mae:.2f})"
    )
    return summary


if __name__ == "__main__":
    # Test the model
    base = Path(__file__).resolve().parents[2]
    df = pd.read_csv(base / "data/processed/cycle_features_with_rul.csv")
    
    batteries = df['battery_id'].unique().tolist()
    n_train = int(len(batteries) * 0.8)
    train_batteries = batteries[:n_train]
    test_batteries = batteries[n_train:]
    
    model, metrics = train_ml_model(
        df, train_batteries, test_batteries,
        base / "trained_models"
    )
    
    logger.info("Metrics:\n%s", json.dumps(metrics.to_dict(), indent=2))

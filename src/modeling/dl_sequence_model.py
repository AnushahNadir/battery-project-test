# src/modeling/dl_sequence_model.py
"""
Sequence model for RUL prediction.

Primary backend:
- Tiny TCN (PyTorch), when torch is installed.

Fallback backend:
- Sequence MLP on flattened sliding windows (scikit-learn).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    HAS_TORCH = False

from src.modeling.ml_model import FEATURE_COLUMNS


if HAS_TORCH:
    class TinyTCN(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_channels, 1)

        def forward(self, x):
            # x: [batch, channels, seq_len]
            h = self.net(x)
            h = h[:, :, -1]
            out = self.head(h)
            return out.squeeze(-1)


@dataclass
class DLSequenceMetrics:
    rmse: float
    mae: float
    battery_generalization_score: float
    per_battery_rmse: Dict[str, float]
    backend: str
    sequence_length: int
    hidden_channels: int

    def to_dict(self) -> Dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "battery_generalization_score": self.battery_generalization_score,
            "per_battery_rmse": self.per_battery_rmse,
            "backend": self.backend,
            "sequence_length": self.sequence_length,
            "hidden_channels": self.hidden_channels,
        }


class TemporalSequenceModel:
    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 8,
        force_backend: str = "auto",
        hidden_channels: int = 32,
        learning_rate: float = 1.0e-3,
        batch_size: int = 64,
        epochs: int = 60,
        patience: int = 10,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
        use_log_target: bool = False,
        huber_beta: float = 3.0,
        grad_clip: float = 1.0,
        lr_factor: float = 0.5,
        lr_patience: int = 4,
    ):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.sequence_length = int(sequence_length)
        self.force_backend = force_backend

        self.backend: str = "untrained"
        self.model_ = None
        self.feature_means_: Optional[np.ndarray] = None
        self.feature_stds_: Optional[np.ndarray] = None
        self.residual_std_: float = 1.0
        self.hidden_channels_: int = int(hidden_channels)
        self.learning_rate_: float = float(learning_rate)
        self.batch_size_: int = int(batch_size)
        self.epochs_: int = int(epochs)
        self.patience_: int = int(patience)
        self.dropout_: float = float(dropout)
        self.weight_decay_: float = float(weight_decay)
        self.use_log_target_: bool = bool(use_log_target)
        self.huber_beta_: float = float(huber_beta)
        self.grad_clip_: float = float(grad_clip)
        self.lr_factor_: float = float(lr_factor)
        self.lr_patience_: int = int(lr_patience)

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        if not self.use_log_target_:
            return y.astype(np.float32)
        y_nonneg = np.maximum(y.astype(float), 0.0)
        return np.log1p(y_nonneg).astype(np.float32)

    def _inverse_target(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.use_log_target_:
            return y_pred.astype(float)
        return np.maximum(np.expm1(y_pred.astype(float)), 0.0)

    def _coerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.reindex(columns=self.feature_columns).copy()
        for col in self.feature_columns:
            s = X[col]
            if s.dtype == object:
                s = s.astype(str).str.strip().str.replace(r"^\[|\]$", "", regex=True)
            X[col] = pd.to_numeric(s, errors="coerce")
        return X

    def _prepare_sequences(
        self,
        df: pd.DataFrame,
        fit_stats: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        work = df.copy().reset_index(drop=True)
        work["__orig_idx"] = np.arange(len(work), dtype=int)
        work["battery_id"] = work["battery_id"].astype(str)
        work["cycle_index"] = pd.to_numeric(work["cycle_index"], errors="coerce")
        work = work.dropna(subset=["battery_id", "cycle_index"]).copy()
        work["cycle_index"] = work["cycle_index"].astype(int)
        work = work.sort_values(["battery_id", "cycle_index"], kind="mergesort").reset_index(drop=True)

        feat_df = self._coerce_features(work)
        feat_arr = feat_df.values.astype(float)

        if fit_stats or self.feature_means_ is None or self.feature_stds_ is None:
            means = np.nanmean(feat_arr, axis=0)
            means = np.where(np.isfinite(means), means, 0.0)
            stds = np.nanstd(feat_arr, axis=0)
            stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0)
            self.feature_means_ = means.astype(float)
            self.feature_stds_ = stds.astype(float)

        means = self.feature_means_
        stds = self.feature_stds_

        # Impute NaN with train means before z-score.
        for j in range(feat_arr.shape[1]):
            col = feat_arr[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = means[j]
            feat_arr[:, j] = col
        feat_arr = (feat_arr - means) / stds

        if "RUL" in work.columns:
            y_arr = pd.to_numeric(work["RUL"], errors="coerce").values.astype(float)
        else:
            y_arr = np.full(len(work), np.nan, dtype=float)

        X_seq = []
        y_seq = []
        orig_idx = []

        for _, g in work.groupby("battery_id", sort=False):
            g_feat = feat_arr[g.index.values]
            g_y = y_arr[g.index.values]
            g_orig = g["__orig_idx"].values.astype(int)

            for i in range(len(g)):
                start = max(0, i - self.sequence_length + 1)
                window = g_feat[start : i + 1]
                if window.shape[0] < self.sequence_length:
                    pad = np.repeat(window[[0]], self.sequence_length - window.shape[0], axis=0)
                    window = np.vstack([pad, window])
                X_seq.append(window.astype(np.float32))
                y_seq.append(float(g_y[i]))
                orig_idx.append(int(g_orig[i]))

        X_seq_np = np.asarray(X_seq, dtype=np.float32)
        y_seq_np = np.asarray(y_seq, dtype=np.float32)
        orig_idx_np = np.asarray(orig_idx, dtype=int)
        return X_seq_np, y_seq_np, orig_idx_np

    def _fit_mlp(self, X_seq: np.ndarray, y: np.ndarray) -> None:
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        y_train = self._transform_target(y)
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            random_state=42,
            max_iter=300,
        )
        model.fit(X_flat, y_train)
        self.model_ = model
        self.backend = "sequence_mlp_fallback"

    def _fit_tcn(self, X_seq: np.ndarray, y: np.ndarray) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available for TCN backend.")

        torch.manual_seed(42)
        n = X_seq.shape[0]
        idx = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(idx)

        n_val = max(1, int(0.1 * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:] if n > n_val else idx

        x_train = torch.tensor(X_seq[train_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_train = torch.tensor(self._transform_target(y[train_idx]), dtype=torch.float32)
        x_val = torch.tensor(X_seq[val_idx].transpose(0, 2, 1), dtype=torch.float32)
        y_val = torch.tensor(self._transform_target(y[val_idx]), dtype=torch.float32)

        model = TinyTCN(
            in_channels=len(self.feature_columns),
            hidden_channels=self.hidden_channels_,
            dropout=self.dropout_,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate_,
            weight_decay=self.weight_decay_,
        )
        loss_fn = nn.SmoothL1Loss(beta=self.huber_beta_)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor_,
            patience=self.lr_patience_,
        )

        best_state = None
        best_val = float("inf")
        patience = self.patience_
        wait = 0
        batch_size = self.batch_size_
        epochs = self.epochs_

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(x_train.shape[0])
            for start in range(0, x_train.shape[0], batch_size):
                batch_ids = perm[start : start + batch_size]
                xb = x_train[batch_ids]
                yb = y_train[batch_ids]
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vpred = model(x_val)
                vloss = loss_fn(vpred, y_val).item()
            scheduler.step(vloss)

            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        self.model_ = model
        self.backend = "torch_tcn"

    def fit(self, df: pd.DataFrame) -> None:
        X_seq, y, _ = self._prepare_sequences(df, fit_stats=True)
        mask = np.isfinite(y)
        X_seq = X_seq[mask]
        y = y[mask]
        if len(y) == 0:
            raise ValueError("No finite RUL targets available for sequence model training.")

        use_tcn = self.force_backend in {"auto", "tcn"} and HAS_TORCH
        if self.force_backend == "mlp":
            use_tcn = False

        if use_tcn:
            self._fit_tcn(X_seq, y)
        else:
            self._fit_mlp(X_seq, y)

        train_preds = self._predict_sequences(X_seq)
        resid = y - train_preds
        std = float(np.nanstd(resid))
        if not np.isfinite(std) or std < 1e-6:
            std = max(float(np.nanstd(y)) * 0.15, 1.0)
        self.residual_std_ = std

    def _predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        if self.backend == "torch_tcn":
            x = torch.tensor(X_seq.transpose(0, 2, 1), dtype=torch.float32)
            with torch.no_grad():
                pred = self.model_(x).cpu().numpy().astype(float)
            return pred

        if self.backend == "sequence_mlp_fallback":
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            return self.model_.predict(X_flat).astype(float)

        raise RuntimeError("Sequence model is not trained.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X_seq, _, orig_idx = self._prepare_sequences(df, fit_stats=False)
        preds_sorted_raw = self._predict_sequences(X_seq)
        preds_sorted = self._inverse_target(preds_sorted_raw)

        out = pd.Series(preds_sorted, index=orig_idx).reindex(range(len(df))).to_numpy(dtype=float)
        if not np.isfinite(out).all():
            finite = out[np.isfinite(out)]
            fill = float(np.nanmedian(finite)) if len(finite) else 0.0
            out = np.where(np.isfinite(out), out, fill)
        return out

    def predict_with_uncertainty(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower_raw, median, upper_raw = self.predict_with_uncertainty_raw(df)
        lower = np.maximum(lower_raw, 0.0)
        upper = np.maximum(upper_raw, median)
        return lower, median, upper

    def predict_with_uncertainty_raw(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict raw uncertainty bounds (unclipped lower/upper).
        """
        median = self.predict(df)
        spread = 1.64 * float(self.residual_std_)
        lower_raw = median - spread
        upper_raw = median + spread
        return lower_raw, median, upper_raw

    def evaluate(self, df: pd.DataFrame) -> DLSequenceMetrics:
        preds = self.predict(df)
        actual = pd.to_numeric(df["RUL"], errors="coerce").values.astype(float)
        mask = np.isfinite(actual) & np.isfinite(preds)
        if not mask.any():
            raise ValueError("No finite rows to evaluate sequence model.")

        preds = preds[mask]
        actual = actual[mask]
        eval_df = df.reset_index(drop=True).loc[np.where(mask)[0]].copy()

        rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
        mae = float(np.mean(np.abs(preds - actual)))

        per_battery = {}
        for bid in eval_df["battery_id"].astype(str).unique():
            m = eval_df["battery_id"].astype(str) == bid
            bpred = preds[m.values]
            bact = actual[m.values]
            per_battery[bid] = float(np.sqrt(np.mean((bpred - bact) ** 2)))

        battery_gen = float(np.mean(list(per_battery.values()))) if per_battery else float("nan")

        return DLSequenceMetrics(
            rmse=rmse,
            mae=mae,
            battery_generalization_score=battery_gen,
            per_battery_rmse=per_battery,
            backend=self.backend,
            sequence_length=self.sequence_length,
            hidden_channels=self.hidden_channels_,
        )

    def save(self, path: Path) -> None:
        payload = {
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "force_backend": self.force_backend,
            "backend": self.backend,
            "feature_means": self.feature_means_,
            "feature_stds": self.feature_stds_,
            "residual_std": self.residual_std_,
            "hidden_channels": self.hidden_channels_,
            "learning_rate": self.learning_rate_,
            "batch_size": self.batch_size_,
            "epochs": self.epochs_,
            "patience": self.patience_,
            "dropout": self.dropout_,
            "weight_decay": self.weight_decay_,
            "use_log_target": self.use_log_target_,
            "huber_beta": self.huber_beta_,
            "grad_clip": self.grad_clip_,
            "lr_factor": self.lr_factor_,
            "lr_patience": self.lr_patience_,
        }

        if self.backend == "torch_tcn":
            payload["state_dict"] = self.model_.state_dict()
        else:
            payload["model"] = self.model_

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "TemporalSequenceModel":
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            feature_columns=data["feature_columns"],
            sequence_length=data["sequence_length"],
            force_backend=data.get("force_backend", "auto"),
            hidden_channels=int(data.get("hidden_channels", 32)),
            learning_rate=float(data.get("learning_rate", 1.0e-3)),
            batch_size=int(data.get("batch_size", 64)),
            epochs=int(data.get("epochs", 60)),
            patience=int(data.get("patience", 10)),
            dropout=float(data.get("dropout", 0.1)),
            weight_decay=float(data.get("weight_decay", 1e-5)),
            use_log_target=bool(data.get("use_log_target", False)),
            huber_beta=float(data.get("huber_beta", 3.0)),
            grad_clip=float(data.get("grad_clip", 1.0)),
            lr_factor=float(data.get("lr_factor", 0.5)),
            lr_patience=int(data.get("lr_patience", 4)),
        )
        obj.backend = data["backend"]
        obj.feature_means_ = np.asarray(data["feature_means"], dtype=float)
        obj.feature_stds_ = np.asarray(data["feature_stds"], dtype=float)
        obj.residual_std_ = float(data.get("residual_std", 1.0))
        obj.hidden_channels_ = int(data.get("hidden_channels", 32))

        if obj.backend == "torch_tcn":
            if not HAS_TORCH:
                raise RuntimeError("Cannot load torch TCN model because PyTorch is unavailable.")
            model = TinyTCN(
                in_channels=len(obj.feature_columns),
                hidden_channels=obj.hidden_channels_,
                dropout=obj.dropout_,
            )
            model.load_state_dict(data["state_dict"])
            model.eval()
            obj.model_ = model
        else:
            obj.model_ = data["model"]

        return obj


def train_dl_sequence_model(
    df: pd.DataFrame,
    train_batteries: List[str],
    test_batteries: List[str],
    output_dir: Path,
    sequence_length: int = 8,
    hidden_channels: int = 32,
    learning_rate: float = 1.0e-3,
    batch_size: int = 64,
    epochs: int = 60,
    patience: int = 10,
    dropout: float = 0.1,
    weight_decay: float = 1e-5,
    force_backend: str = "auto",
    use_log_target: bool = False,
    huber_beta: float = 3.0,
    grad_clip: float = 1.0,
    lr_factor: float = 0.5,
    lr_patience: int = 4,
) -> Tuple[TemporalSequenceModel, DLSequenceMetrics]:
    train_df = df[df["battery_id"].isin(train_batteries)].copy()
    test_df = df[df["battery_id"].isin(test_batteries)].copy()

    logger.info(f"Training sequence model on {len(train_batteries)} batteries...")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")

    model = TemporalSequenceModel(
        sequence_length=sequence_length,
        force_backend=force_backend,
        hidden_channels=hidden_channels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        dropout=dropout,
        weight_decay=weight_decay,
        use_log_target=use_log_target,
        huber_beta=huber_beta,
        grad_clip=grad_clip,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
    )
    model.fit(train_df)
    metrics = model.evaluate(test_df)

    logger.info(f"  Backend: {metrics.backend}")
    logger.info(f"  RMSE: {metrics.rmse:.2f} cycles")
    logger.info(f"  MAE: {metrics.mae:.2f} cycles")
    logger.info(f"  Battery Generalization: {metrics.battery_generalization_score:.2f}")
    logger.info(
        f"  Config: seq_len={sequence_length}, hidden={hidden_channels}, "
        f"lr={learning_rate}, epochs={epochs}, batch={batch_size}, dropout={dropout}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dl_sequence_model.pkl"
    model.save(model_path)
    logger.info(f"  Saved model to {model_path}")

    return model, metrics

"""
2-layer LSTM baseline for RUL prediction.

Same train/predict/evaluate interface as TemporalSequenceModel so it can be
dropped into benchmark_multi_seed.py without changes to the harness.

Requires PyTorch. Falls back to a scikit-learn MLPRegressor on flattened
windows when torch is unavailable (matching the TCN fallback pattern).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAS_TORCH = False

from src.modeling.ml_model import FEATURE_COLUMNS


@dataclass
class LSTMMetrics:
    rmse: float
    mae: float
    battery_generalization_score: float
    per_battery_rmse: Dict[str, float]
    backend: str
    sequence_length: int
    hidden_size: int

    def to_dict(self) -> Dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "battery_generalization_score": self.battery_generalization_score,
            "per_battery_rmse": self.per_battery_rmse,
            "backend": self.backend,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
        }


if HAS_TORCH:
    class _TwoLayerLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0.0,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for _bid, g in df.groupby("battery_id", sort=False):
        g = g.sort_values("cycle_index", kind="mergesort").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=float)
        rul = pd.to_numeric(g["RUL"], errors="coerce").to_numpy(dtype=float)
        n = len(g)
        for i in range(n):
            if not np.isfinite(rul[i]):
                continue
            start = max(0, i - seq_len + 1)
            window = feats[start: i + 1]
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), len(feature_cols)), dtype=float)
                window = np.vstack([pad, window])
            X_list.append(window)
            y_list.append(rul[i])
    if not X_list:
        raise ValueError("No valid sequences from training dataframe.")
    return np.stack(X_list, axis=0).astype(np.float32), np.array(y_list, dtype=np.float32)


def _standardize(
    df: pd.DataFrame,
    feature_cols: List[str],
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = 0.0
    arr = out[feature_cols].to_numpy(dtype=float)
    if means is None:
        means = np.nanmean(arr, axis=0)
    if stds is None:
        stds = np.nanstd(arr, axis=0)
        stds = np.where(stds < 1e-9, 1.0, stds)
    arr = np.where(np.isfinite(arr), arr, means)
    arr = (arr - means) / stds
    out[feature_cols] = arr
    return out, means, stds


def _windows_for_df(
    df: pd.DataFrame, feature_cols: List[str], seq_len: int
) -> Tuple[np.ndarray, List[int]]:
    """Build prediction windows preserving original row order."""
    windows: List[np.ndarray] = []
    orig_indices: List[int] = []
    for _bid, g in df.groupby("battery_id", sort=False):
        g_sorted = g.sort_values("cycle_index", kind="mergesort")
        feats = g_sorted[feature_cols].to_numpy(dtype=np.float32)
        n = len(g_sorted)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            window = feats[start: i + 1]
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), len(feature_cols)), dtype=np.float32)
                window = np.vstack([pad, window])
            windows.append(window)
            orig_indices.append(int(g_sorted.iloc[i].name))
    return np.stack(windows, axis=0), orig_indices


class LSTMRULModel:
    """
    2-layer LSTM RUL predictor. Same interface as TemporalSequenceModel.
    Uses PyTorch LSTM when available; sklearn MLP fallback otherwise.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 8,
        hidden_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        patience: int = 8,
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
    ):
        self.feature_columns = feature_columns or list(FEATURE_COLUMNS)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.weight_decay = weight_decay

        self._model: Optional["_TwoLayerLSTM"] = None
        self._mlp = None
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._backend: str = "unfit"

    def fit(self, df: pd.DataFrame) -> None:
        df_std, self._means, self._stds = _standardize(df, self.feature_columns)
        if HAS_TORCH:
            self._fit_torch(df_std)
        else:
            self._fit_mlp(df_std)

    def _fit_torch(self, df: pd.DataFrame) -> None:
        X, y = _build_sequences(df, self.feature_columns, self.sequence_length)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _TwoLayerLSTM(len(self.feature_columns), self.hidden_size, self.dropout).to(device)
        dataset = TensorDataset(torch.tensor(X, device=device), torch.tensor(y, device=device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
        loss_fn = nn.HuberLoss(delta=3.0)

        best_loss, best_state, wait = float("inf"), None, 0
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(dataset)
            scheduler.step(epoch_loss)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("[LSTM] Early stop epoch %d (loss=%.3f)", epoch + 1, best_loss)
                    break
        if best_state:
            model.load_state_dict(best_state)
        self._model = model.cpu()
        self._backend = f"torch_lstm_2layer"

    def _fit_mlp(self, df: pd.DataFrame) -> None:
        from sklearn.neural_network import MLPRegressor
        X, y = _build_sequences(df, self.feature_columns, self.sequence_length)
        self._mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                                 random_state=42, early_stopping=True)
        self._mlp.fit(X.reshape(len(X), -1), y)
        self._backend = "sklearn_mlp_fallback"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._means is None:
            raise RuntimeError("Call fit() before predict().")
        df_std, _, _ = _standardize(df, self.feature_columns, self._means, self._stds)

        n_rows = len(df)
        preds_out = np.full(n_rows, np.nan, dtype=float)
        windows, orig_indices = _windows_for_df(df_std, self.feature_columns, self.sequence_length)

        if HAS_TORCH and self._model is not None:
            self._model.eval()
            xt = torch.tensor(windows, dtype=torch.float32)
            with torch.no_grad():
                raw = self._model(xt).numpy()
        else:
            raw = self._mlp.predict(windows.reshape(len(windows), -1))

        # Map predictions back to original row positions.
        df_reset = df.reset_index(drop=True)
        for pred_i, orig_idx in enumerate(orig_indices):
            row_pos = df_reset.index.get_loc(orig_idx) if orig_idx in df_reset.index else pred_i
            if row_pos < n_rows:
                preds_out[row_pos] = max(0.0, float(raw[pred_i]))

        # Fill any unmapped positions with 0.
        preds_out = np.where(np.isfinite(preds_out), preds_out, 0.0)
        return preds_out

    def evaluate(self, df: pd.DataFrame) -> LSTMMetrics:
        preds = self.predict(df)
        y_true = pd.to_numeric(df["RUL"], errors="coerce").to_numpy(float)
        valid = np.isfinite(y_true) & np.isfinite(preds)

        rmse = float(np.sqrt(np.mean((preds[valid] - y_true[valid]) ** 2))) if valid.any() else float("nan")
        mae = float(np.mean(np.abs(preds[valid] - y_true[valid]))) if valid.any() else float("nan")

        per_battery: Dict[str, float] = {}
        bgs_vals: List[float] = []
        df_r = df.reset_index(drop=True)
        for bid, g in df_r.groupby("battery_id", sort=False):
            p = preds[g.index]
            yt = pd.to_numeric(g["RUL"], errors="coerce").to_numpy(float)
            v = np.isfinite(yt) & np.isfinite(p)
            if v.any():
                br = float(np.sqrt(np.mean((p[v] - yt[v]) ** 2)))
                per_battery[str(bid)] = round(br, 4)
                bgs_vals.append(br)

        return LSTMMetrics(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            battery_generalization_score=round(float(np.mean(bgs_vals)) if bgs_vals else float("nan"), 4),
            per_battery_rmse=per_battery,
            backend=self._backend,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
        )

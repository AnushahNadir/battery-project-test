"""
Property-based invariants for the battery RUL pipeline.

Run:
    python -m pytest tests/test_invariants.py -v
    python -m pytest tests/test_invariants.py -v --hypothesis-seed=0
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, note, settings, strategies as st
from scipy.stats import norm


settings.register_profile("ci", max_examples=200, deadline=5000)
settings.load_profile("ci")


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_drift():
    from src.pipeline.drift import (
        CONDITION_FEATURES,
        DEGRADATION_FEATURES,
        DriftMonitor,
        FeatureDistribution,
        PSI_AMBER,
        PSI_RED,
        _compute_psi,
    )

    return DriftMonitor, FeatureDistribution, _compute_psi, CONDITION_FEATURES, DEGRADATION_FEATURES, PSI_AMBER, PSI_RED


def _import_uncertainty():
    from src.modeling.uncertainty import DEFAULT_HORIZON, UncertaintyQuantifier

    # We only need a thin wrapper around methods for direct property testing.
    uq = UncertaintyQuantifier(stat_model=object(), ml_model=object())

    return (
        lambda rp, lo, hi, h: uq.compute_failure_probability(rp, lo, hi, horizon=h),
        lambda p: uq.categorize_risk(p),
        DEFAULT_HORIZON,
    )


def _import_config():
    from pydantic import ValidationError

    from src.config import DLModelConfig, PipelineConfig, RangeConfig, RiskConfig, UncertaintyConfig

    return PipelineConfig, RangeConfig, RiskConfig, UncertaintyConfig, DLModelConfig, ValidationError


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

finite_array = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=5,
    max_size=400,
).map(lambda x: np.asarray(x, dtype=float))

probability = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ============================================================================
# 1) PSI math invariants (4)
# ============================================================================


class TestPSIMath:
    @given(data=finite_array)
    @settings(max_examples=250)
    def test_psi_self_is_near_zero(self, data: np.ndarray):
        _, FeatureDistribution, _compute_psi, *_ = _import_drift()
        assume = pytest.assume if hasattr(pytest, "assume") else None
        if len(data) < 5 or np.std(data) <= 0:
            return
        dist = FeatureDistribution.fit("x", data)
        psi = _compute_psi(data, data, np.asarray(dist.bin_edges, dtype=float))
        note(f"psi_self={psi:.6f}, std={np.std(data):.4f}")
        assert psi >= -1e-9
        assert psi < 0.05

    @given(expected=finite_array, actual=finite_array)
    @settings(max_examples=250)
    def test_psi_is_non_negative(self, expected: np.ndarray, actual: np.ndarray):
        _, FeatureDistribution, _compute_psi, *_ = _import_drift()
        if np.std(expected) <= 0:
            return
        dist = FeatureDistribution.fit("x", expected)
        psi = _compute_psi(expected, actual, np.asarray(dist.bin_edges, dtype=float))
        assert np.isfinite(psi)
        assert psi >= -1e-9

    @given(
        mean=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
        shift_small=st.floats(min_value=0.05, max_value=1.0, allow_nan=False, allow_infinity=False),
        shift_large=st.floats(min_value=1.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_psi_monotone_in_shift_for_gaussian(
        self, mean: float, std: float, shift_small: float, shift_large: float
    ):
        _, FeatureDistribution, _compute_psi, *_ = _import_drift()
        # Use averaged PSI across multiple draws to reduce finite-sample noise.
        rng = np.random.default_rng(123)
        psi_small_vals = []
        psi_large_vals = []
        for _ in range(12):
            expected = rng.normal(loc=mean, scale=std, size=1200)
            actual_small = rng.normal(loc=mean + shift_small, scale=std, size=1200)
            actual_large = rng.normal(loc=mean + shift_large, scale=std, size=1200)
            dist = FeatureDistribution.fit("x", expected)
            edges = np.asarray(dist.bin_edges, dtype=float)
            psi_small_vals.append(_compute_psi(expected, actual_small, edges))
            psi_large_vals.append(_compute_psi(expected, actual_large, edges))

        psi_small = float(np.mean(psi_small_vals))
        psi_large = float(np.mean(psi_large_vals))
        note(f"mean_psi_small={psi_small:.4f}, mean_psi_large={psi_large:.4f}")
        assert psi_large >= psi_small - 1e-6

    @given(expected=finite_array, actual=finite_array)
    @settings(max_examples=200)
    def test_psi_is_finite(self, expected: np.ndarray, actual: np.ndarray):
        _, FeatureDistribution, _compute_psi, *_ = _import_drift()
        if np.std(expected) <= 0:
            return
        dist = FeatureDistribution.fit("x", expected)
        psi = _compute_psi(expected, actual, np.asarray(dist.bin_edges, dtype=float))
        assert np.isfinite(psi)


# ============================================================================
# 2) Uncertainty / risk invariants (5)
# ============================================================================


class TestUncertaintyRisk:
    @given(
        rul_pred=st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False),
        lower=st.floats(min_value=-150, max_value=200, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=-100, max_value=400, allow_nan=False, allow_infinity=False),
        horizon=st.integers(min_value=1, max_value=120),
    )
    @settings(max_examples=400)
    def test_failure_probability_in_unit_interval(self, rul_pred, lower, upper, horizon):
        fp_fn, *_ = _import_uncertainty()
        if lower >= upper:
            return
        fp = fp_fn(rul_pred, lower, upper, horizon)
        assert 0.0 <= fp <= 1.0

    @given(p=probability)
    @settings(max_examples=250)
    def test_risk_category_values(self, p):
        _, risk_fn, _ = _import_uncertainty()
        out = risk_fn(p)
        assert out in {"LOW", "MEDIUM", "HIGH"}

    @given(
        base=st.floats(min_value=0, max_value=120, allow_nan=False, allow_infinity=False),
        span=st.floats(min_value=10, max_value=120, allow_nan=False, allow_infinity=False),
        horizon=st.integers(min_value=1, max_value=60),
    )
    @settings(max_examples=250)
    def test_higher_rul_reduces_failure_probability(self, base, span, horizon):
        fp_fn, *_ = _import_uncertainty()
        lo1, hi1 = base - span / 2, base + span / 2
        lo2, hi2 = lo1 + 40, hi1 + 40
        p1 = fp_fn(base, lo1, hi1, horizon)
        p2 = fp_fn(base + 40, lo2, hi2, horizon)
        assert p1 >= p2 - 1e-9

    @given(
        rul_pred=st.floats(min_value=30, max_value=180, allow_nan=False, allow_infinity=False),
        horizon=st.integers(min_value=1, max_value=30),
        width=st.floats(min_value=5.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=250)
    def test_wider_interval_increases_failure_probability_for_healthy_battery(self, rul_pred, horizon, width):
        fp_fn, *_ = _import_uncertainty()
        if rul_pred <= 2 * horizon:
            return
        lo_n = rul_pred - width * 0.5
        hi_n = rul_pred + width * 0.5
        lo_w = rul_pred - width * 2.0
        hi_w = rul_pred + width * 2.0
        p_n = fp_fn(rul_pred, lo_n, hi_n, horizon)
        p_w = fp_fn(rul_pred, lo_w, hi_w, horizon)
        assert p_w >= p_n - 1e-9

    def test_risk_threshold_boundaries(self):
        from src.config import get_config

        _, risk_fn, _ = _import_uncertainty()
        cfg = get_config()
        med = cfg.risk.medium_threshold
        high = cfg.risk.high_threshold

        assert risk_fn(max(0.0, med - 1e-6)) == "LOW"
        assert risk_fn(med) == "MEDIUM"
        assert risk_fn(max(med, high - 1e-6)) in {"MEDIUM", "HIGH"}
        assert risk_fn(high) == "HIGH"


# ============================================================================
# 3) Conformal calibration invariants (4)
# ============================================================================


class TestConformal:
    @given(rul_pred=st.floats(min_value=0, max_value=250), n_rows=st.integers(min_value=1, max_value=120))
    @settings(max_examples=200)
    def test_conformal_lower_bound_non_negative(self, rul_pred, n_rows):
        from src.modeling.conformal import ConformalCalibrator, GroupCalibration

        cal = ConformalCalibrator(coverage=0.90)
        cal._fitted = True
        q = abs(rul_pred) + 10.0
        cal._global_q_hat = q
        cal._group_cal["room"] = GroupCalibration(
            group="room",
            n_cal=100,
            n_cal_batteries=2,
            scores=[q],
            q_hat=q,
            coverage=0.90,
            strategy="split",
            inflated=False,
        )

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                return np.full(len(df), rul_pred, dtype=float)

        df = pd.DataFrame({"battery_id": ["B0005"] * n_rows, "cycle_index": range(n_rows), "x": 0.0})
        out = cal.predict(df, MockModel())
        assert (out["rul_lower_conformal"].to_numpy(dtype=float) >= 0).all()

    @given(n_rows=st.integers(min_value=1, max_value=300))
    @settings(max_examples=120)
    def test_conformal_output_length_preserved(self, n_rows):
        from src.modeling.conformal import ConformalCalibrator, GroupCalibration

        cal = ConformalCalibrator(coverage=0.90)
        cal._fitted = True
        cal._global_q_hat = 25.0
        cal._group_cal["room"] = GroupCalibration(
            group="room",
            n_cal=50,
            n_cal_batteries=2,
            scores=[1.0, 2.0, 3.0],
            q_hat=25.0,
            coverage=0.90,
            strategy="split",
            inflated=False,
        )

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                return np.full(len(df), 30.0, dtype=float)

        df = pd.DataFrame({"battery_id": ["B0005"] * n_rows, "cycle_index": range(n_rows), "x": 0.0})
        out = cal.predict(df, MockModel())
        assert len(out) == n_rows

    @given(
        q_hat=st.floats(min_value=0.1, max_value=200.0, allow_nan=False, allow_infinity=False),
        pred=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=250)
    def test_conformal_interval_width_consistent(self, q_hat, pred):
        from src.modeling.conformal import ConformalCalibrator, GroupCalibration

        cal = ConformalCalibrator(coverage=0.90)
        cal._fitted = True
        cal._global_q_hat = q_hat
        cal._group_cal["room"] = GroupCalibration(
            group="room",
            n_cal=10,
            n_cal_batteries=2,
            scores=[q_hat],
            q_hat=q_hat,
            coverage=0.90,
            strategy="split",
            inflated=False,
        )

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                return np.asarray([pred], dtype=float)

        df = pd.DataFrame({"battery_id": ["B0005"], "cycle_index": [1], "x": [0.0]})
        out = cal.predict(df, MockModel())
        lo = float(out["rul_lower_conformal"].iloc[0])
        hi = float(out["rul_upper_conformal"].iloc[0])
        width = float(out["interval_width"].iloc[0])
        assert hi >= lo
        assert width >= 0
        assert abs(width - (hi - lo)) < 1e-9

    def test_conformal_save_load_round_trip(self):
        from src.modeling.conformal import ConformalCalibrator, GroupCalibration

        cal = ConformalCalibrator(coverage=0.90)
        cal._fitted = True
        cal._global_q_hat = 42.5
        cal._group_cal["room"] = GroupCalibration(
            group="room",
            n_cal=100,
            n_cal_batteries=3,
            scores=list(np.linspace(5, 50, 100)),
            q_hat=33.3,
            coverage=0.90,
            strategy="split",
            inflated=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            p = Path(tf.name)
        cal.save(p)
        loaded = ConformalCalibrator.load(p)
        p.unlink(missing_ok=True)

        assert abs(loaded._global_q_hat - cal._global_q_hat) < 1e-9
        assert abs(loaded._group_cal["room"].q_hat - cal._group_cal["room"].q_hat) < 1e-9


# ============================================================================
# 4) Drift monitor invariants (4)
# ============================================================================


class TestDriftMonitor:
    def _fit_monitor(self, n: int = 400, seed: int = 42):
        DriftMonitor, *_ = _import_drift()
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {
                "battery_id": [f"B{i // 20:04d}" for i in range(n)],
                "temp_mean": rng.uniform(20, 28, n),
                "temp_max": rng.uniform(25, 35, n),
                "v_min": rng.uniform(2.5, 2.8, n),
                "v_mean": rng.uniform(3.2, 3.7, n),
                "i_mean": rng.uniform(1.5, 2.5, n),
                "i_min": rng.uniform(0.8, 1.5, n),
                "duration_s": rng.uniform(2800, 3600, n),
                "capacity": rng.uniform(1.2, 2.0, n),
                "ah_est": rng.uniform(1.2, 2.0, n),
                "energy_j": rng.uniform(15000, 25000, n),
            }
        )
        return DriftMonitor.fit(df)

    @given(n_rows=st.integers(min_value=10, max_value=250))
    @settings(max_examples=120)
    def test_report_row_count_matches_input(self, n_rows):
        monitor = self._fit_monitor()
        rng = np.random.default_rng(n_rows)
        df = pd.DataFrame(
            {
                "battery_id": ["B0000"] * n_rows,
                "temp_mean": rng.uniform(20, 28, n_rows),
                "temp_max": rng.uniform(25, 35, n_rows),
                "v_min": rng.uniform(2.5, 2.8, n_rows),
                "v_mean": rng.uniform(3.2, 3.7, n_rows),
                "i_mean": rng.uniform(1.5, 2.5, n_rows),
                "i_min": rng.uniform(0.8, 1.5, n_rows),
                "duration_s": rng.uniform(2800, 3600, n_rows),
                "capacity": rng.uniform(1.2, 2.0, n_rows),
                "ah_est": rng.uniform(1.2, 2.0, n_rows),
                "energy_j": rng.uniform(15000, 25000, n_rows),
            }
        )
        report = monitor.compute(df)
        assert report.n_rows == n_rows

    @given(n_rows=st.integers(min_value=10, max_value=250))
    @settings(max_examples=120)
    def test_all_psi_values_non_negative(self, n_rows):
        monitor = self._fit_monitor()
        rng = np.random.default_rng(n_rows + 17)
        df = pd.DataFrame(
            {
                "battery_id": ["B0000"] * n_rows,
                "temp_mean": rng.uniform(10, 50, n_rows),
                "temp_max": rng.uniform(15, 60, n_rows),
                "v_min": rng.uniform(2.0, 4.0, n_rows),
                "v_mean": rng.uniform(2.5, 4.5, n_rows),
                "i_mean": rng.uniform(0.5, 6.0, n_rows),
                "i_min": rng.uniform(0.2, 4.0, n_rows),
                "duration_s": rng.uniform(500, 8000, n_rows),
                "capacity": rng.uniform(0.5, 2.5, n_rows),
                "ah_est": rng.uniform(0.5, 2.5, n_rows),
                "energy_j": rng.uniform(3000, 35000, n_rows),
            }
        )
        report = monitor.compute(df)
        for feat in report.features:
            if np.isfinite(feat.psi):
                assert feat.psi >= -1e-9

    @given(n_rows=st.integers(min_value=10, max_value=200))
    @settings(max_examples=120)
    def test_status_values_valid(self, n_rows):
        monitor = self._fit_monitor()
        rng = np.random.default_rng(n_rows + 99)
        df = pd.DataFrame(
            {
                "battery_id": ["B0001"] * n_rows,
                "temp_mean": rng.uniform(10, 50, n_rows),
                "temp_max": rng.uniform(15, 60, n_rows),
                "v_min": rng.uniform(2.0, 4.0, n_rows),
                "v_mean": rng.uniform(2.5, 4.5, n_rows),
                "i_mean": rng.uniform(0.5, 6.0, n_rows),
                "i_min": rng.uniform(0.2, 4.0, n_rows),
                "duration_s": rng.uniform(500, 8000, n_rows),
                "capacity": rng.uniform(0.5, 2.5, n_rows),
                "ah_est": rng.uniform(0.5, 2.5, n_rows),
                "energy_j": rng.uniform(3000, 35000, n_rows),
            }
        )
        report = monitor.compute(df)
        valid = {"GREEN", "AMBER", "RED", "UNKNOWN"}
        assert report.overall_status in valid
        for feat in report.features:
            assert feat.status in valid

    def test_drift_save_load_round_trip(self):
        monitor = self._fit_monitor()
        rng = np.random.default_rng(777)
        test_df = pd.DataFrame(
            {
                "battery_id": ["B0001"] * 60,
                "temp_mean": rng.uniform(22, 32, 60),
                "temp_max": rng.uniform(28, 40, 60),
                "v_min": rng.uniform(2.4, 2.9, 60),
                "v_mean": rng.uniform(3.0, 3.8, 60),
                "i_mean": rng.uniform(1.2, 3.0, 60),
                "i_min": rng.uniform(0.6, 2.0, 60),
                "duration_s": rng.uniform(2500, 4000, 60),
                "capacity": rng.uniform(1.0, 1.9, 60),
                "ah_est": rng.uniform(1.0, 1.9, 60),
                "energy_j": rng.uniform(12000, 22000, 60),
            }
        )

        rep1 = monitor.compute(test_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            p = Path(tf.name)
        monitor.save(p)
        loaded = _import_drift()[0].load(p)
        p.unlink(missing_ok=True)
        rep2 = loaded.compute(test_df)

        psi1 = {f.feature: f.psi for f in rep1.features}
        psi2 = {f.feature: f.psi for f in rep2.features}
        assert psi1.keys() == psi2.keys()
        for k in psi1:
            assert np.isclose(psi1[k], psi2[k], atol=1e-9)


# ============================================================================
# 5) Config validation invariants (5)
# ============================================================================


class TestConfigValidation:
    @given(
        medium=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        high=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=250)
    def test_risk_threshold_ordering_enforced(self, medium, high):
        _, _, RiskConfig, _, _, ValidationError = _import_config()
        if medium < high:
            cfg = RiskConfig(medium_threshold=medium, high_threshold=high)
            assert cfg.medium_threshold < cfg.high_threshold
        else:
            with pytest.raises(ValidationError):
                RiskConfig(medium_threshold=medium, high_threshold=high)

    @given(
        q_low=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        q_high=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=250)
    def test_quantile_ordering_enforced(self, q_low, q_high):
        _, _, _, UncertaintyConfig, _, ValidationError = _import_config()
        is_valid = (q_low < q_high) and (q_low < 0.5) and (q_high > 0.5)
        if is_valid:
            cfg = UncertaintyConfig(q_low=q_low, q_high=q_high)
            assert cfg.q_low < cfg.q_high
        else:
            with pytest.raises(ValidationError):
                UncertaintyConfig(q_low=q_low, q_high=q_high)

    @given(
        patience=st.integers(min_value=1, max_value=200),
        epochs=st.integers(min_value=5, max_value=200),
    )
    @settings(max_examples=250)
    def test_patience_lt_epochs_enforced(self, patience, epochs):
        _, _, _, _, DLModelConfig, ValidationError = _import_config()
        if patience < epochs:
            cfg = DLModelConfig(patience=patience, epochs=epochs)
            assert cfg.patience < cfg.epochs
        else:
            with pytest.raises(ValidationError):
                DLModelConfig(patience=patience, epochs=epochs)

    @given(
        min_val=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        max_val=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=250)
    def test_range_config_min_lt_max_enforced(self, min_val, max_val):
        _, RangeConfig, _, _, _, ValidationError = _import_config()
        if min_val < max_val:
            cfg = RangeConfig(min=min_val, max=max_val)
            assert cfg.min < cfg.max
        else:
            with pytest.raises(ValidationError):
                RangeConfig(min=min_val, max=max_val)

    @given(
        risk_h=st.integers(min_value=1, max_value=500),
        surv_h=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=200)
    def test_horizon_consistency_enforced(self, risk_h, surv_h):
        PipelineConfig, *_rest, ValidationError = _import_config()
        if risk_h == surv_h:
            cfg = PipelineConfig(
                risk={"horizon_cycles": risk_h},
                survival={"horizon_cycles": surv_h},
            )
            assert cfg.risk.horizon_cycles == cfg.survival.horizon_cycles
        else:
            with pytest.raises(ValidationError):
                PipelineConfig(
                    risk={"horizon_cycles": risk_h},
                    survival={"horizon_cycles": surv_h},
                )


# ============================================================================
# 6) RUL label invariants (4)
# ============================================================================


class TestRULLabels:
    @given(n_cycles=st.integers(min_value=2, max_value=300))
    @settings(max_examples=250)
    def test_rul_non_negative(self, n_cycles):
        rul = [max(0, n_cycles - 1 - i) for i in range(n_cycles)]
        assert min(rul) >= 0

    @given(n_cycles=st.integers(min_value=2, max_value=300))
    @settings(max_examples=250)
    def test_rul_last_is_zero(self, n_cycles):
        rul = [max(0, n_cycles - 1 - i) for i in range(n_cycles)]
        assert rul[-1] == 0

    @given(n_cycles=st.integers(min_value=2, max_value=300))
    @settings(max_examples=250)
    def test_rul_first_is_n_minus_one(self, n_cycles):
        rul = [max(0, n_cycles - 1 - i) for i in range(n_cycles)]
        assert rul[0] == n_cycles - 1

    @given(n_cycles=st.integers(min_value=2, max_value=300))
    @settings(max_examples=250)
    def test_rul_non_increasing(self, n_cycles):
        rul = [max(0, n_cycles - 1 - i) for i in range(n_cycles)]
        assert all(rul[i] <= rul[i - 1] for i in range(1, len(rul)))


# ============================================================================
# 7) Output schema invariants (3)
# ============================================================================


class TestOutputSchema:
    REQUIRED_CONFORMAL_COLS = {
        "battery_id",
        "cycle_index",
        "rul_pred",
        "rul_lower_conformal",
        "rul_upper_conformal",
        "q_hat",
        "interval_width",
        "failure_probability",
        "risk_category",
    }

    def _mock_calibrator(self):
        from src.modeling.conformal import ConformalCalibrator, GroupCalibration

        cal = ConformalCalibrator(coverage=0.90)
        cal._fitted = True
        cal._global_q_hat = 30.0
        cal._group_cal["room"] = GroupCalibration(
            group="room",
            n_cal=50,
            n_cal_batteries=2,
            scores=[1.0, 2.0, 3.0],
            q_hat=30.0,
            coverage=0.90,
            strategy="split",
            inflated=False,
        )
        return cal

    @given(n_rows=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_conformal_output_required_columns(self, n_rows):
        cal = self._mock_calibrator()

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                return np.full(len(df), 25.0, dtype=float)

        df = pd.DataFrame({"battery_id": ["B0005"] * n_rows, "cycle_index": range(n_rows), "x": 0.0})
        out = cal.predict(df, MockModel())
        missing = self.REQUIRED_CONFORMAL_COLS - set(out.columns)
        assert not missing

    @given(n_rows=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_conformal_failure_probability_in_unit_interval(self, n_rows):
        cal = self._mock_calibrator()

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                rng = np.random.default_rng(n_rows)
                return rng.uniform(0, 150, len(df))

        df = pd.DataFrame({"battery_id": ["B0005"] * n_rows, "cycle_index": range(n_rows), "x": 0.0})
        out = cal.predict(df, MockModel(), horizon=20)
        p = out["failure_probability"].to_numpy(dtype=float)
        assert (p >= 0).all() and (p <= 1).all()

    @given(n_rows=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_conformal_upper_gte_lower(self, n_rows):
        cal = self._mock_calibrator()

        class MockModel:
            feature_columns = ["x"]

            def predict(self, df):
                rng = np.random.default_rng(n_rows * 3)
                return rng.uniform(0, 200, len(df))

        df = pd.DataFrame({"battery_id": ["B0005"] * n_rows, "cycle_index": range(n_rows), "x": 0.0})
        out = cal.predict(df, MockModel())
        lo = out["rul_lower_conformal"].to_numpy(dtype=float)
        hi = out["rul_upper_conformal"].to_numpy(dtype=float)
        assert (hi >= lo).all()

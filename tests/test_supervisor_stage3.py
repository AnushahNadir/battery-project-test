"""Tests for SupervisorReviewer Stage 3 RMSE verdict logic."""
import json
import math
import pytest
from pathlib import Path

from src.modeling.supervisor_review import SupervisorReviewer


def make_reviewer(
    tmp_path: Path,
    ml_rmse: float,
    threshold: float,
    cv_mean_rmse: float | None = None,
    cv_std_rmse: float = 0.0,
    cv_ratio_warn: float = 1.5,
) -> SupervisorReviewer:
    """Create a SupervisorReviewer with a minimal model_metrics.json."""
    trained_models_dir = tmp_path / "trained_models"
    trained_models_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    metrics: dict = {"rmse": ml_rmse}
    if cv_mean_rmse is not None:
        metrics["groupkfold_cv"] = {"mean_rmse": cv_mean_rmse, "std_rmse": cv_std_rmse}

    (trained_models_dir / "model_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    return SupervisorReviewer(
        output_dir,
        trained_models_dir=trained_models_dir,
        rmse_pass_threshold=threshold,
        cv_rmse_ratio_warn_threshold=cv_ratio_warn,
    )


class TestStage3RmseVerdict:
    def test_pass_when_rmse_below_threshold(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=23.1, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - PASS" in report

    def test_conditional_pass_when_rmse_between_threshold_and_2x(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=150.0, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - CONDITIONAL PASS" in report

    def test_fail_when_rmse_above_2x_threshold(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=250.0, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - FAIL" in report

    def test_fail_when_model_metrics_missing(self, tmp_path):
        """Missing model_metrics.json should default rmse to inf → FAIL."""
        trained_models_dir = tmp_path / "trained_models"
        trained_models_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        reviewer = SupervisorReviewer(output_dir, trained_models_dir=trained_models_dir, rmse_pass_threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - FAIL" in report

    def test_report_shows_rmse_and_threshold(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=23.1, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "23.10 cycles" in report
        assert "threshold: 100 cycles" in report

    def test_boundary_exactly_at_threshold_is_pass(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=100.0, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - PASS" in report

    def test_boundary_exactly_at_2x_threshold_is_conditional_pass(self, tmp_path):
        reviewer = make_reviewer(tmp_path, ml_rmse=200.0, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - CONDITIONAL PASS" in report


class TestStage3CvGap:
    def test_pass_downgraded_to_conditional_when_cv_gap_large(self, tmp_path):
        """RMSE well within threshold but CV gap >1.5x → CONDITIONAL PASS."""
        reviewer = make_reviewer(
            tmp_path, ml_rmse=23.1, threshold=100.0,
            cv_mean_rmse=47.5, cv_std_rmse=16.0, cv_ratio_warn=1.5,
        )
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - CONDITIONAL PASS" in report

    def test_warning_text_present_when_cv_gap_large(self, tmp_path):
        reviewer = make_reviewer(
            tmp_path, ml_rmse=23.1, threshold=100.0,
            cv_mean_rmse=47.5, cv_std_rmse=16.0, cv_ratio_warn=1.5,
        )
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "WARNING" in report
        assert "CV/test RMSE ratio" in report
        assert "test batteries may be unrepresentatively easy" in report

    def test_no_warning_when_cv_gap_acceptable(self, tmp_path):
        """CV RMSE only 1.2x test RMSE → no warning, verdict stays PASS."""
        reviewer = make_reviewer(
            tmp_path, ml_rmse=40.0, threshold=100.0,
            cv_mean_rmse=48.0, cv_std_rmse=5.0, cv_ratio_warn=1.5,
        )
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - PASS" in report
        assert "WARNING" not in report

    def test_cv_rmse_line_shown_even_without_warning(self, tmp_path):
        reviewer = make_reviewer(
            tmp_path, ml_rmse=40.0, threshold=100.0,
            cv_mean_rmse=48.0, cv_std_rmse=5.0, cv_ratio_warn=1.5,
        )
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "GroupKFold CV RMSE: 48.00" in report

    def test_no_cv_line_when_cv_data_absent(self, tmp_path):
        """If groupkfold_cv is missing from metrics, no CV line should appear."""
        reviewer = make_reviewer(tmp_path, ml_rmse=23.1, threshold=100.0)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "GroupKFold CV RMSE" not in report

    def test_fail_verdict_not_further_downgraded_by_cv_gap(self, tmp_path):
        """A FAIL from bad RMSE should stay FAIL regardless of CV gap."""
        reviewer = make_reviewer(
            tmp_path, ml_rmse=250.0, threshold=100.0,
            cv_mean_rmse=400.0, cv_std_rmse=20.0, cv_ratio_warn=1.5,
        )
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 3: Modeling - FAIL" in report


class TestStage6Verdict:
    def _make(self, tmp_path, with_anomalies_file: bool, anomalies_content=None):
        trained_models_dir = tmp_path / "trained_models"
        trained_models_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (trained_models_dir / "model_metrics.json").write_text(
            json.dumps({"rmse": 23.1}), encoding="utf-8"
        )
        if with_anomalies_file:
            content = anomalies_content if anomalies_content is not None else []
            (output_dir / "anomalies.json").write_text(
                json.dumps(content), encoding="utf-8"
            )
        return SupervisorReviewer(
            output_dir, trained_models_dir=trained_models_dir, rmse_pass_threshold=100.0
        )

    def test_pass_when_anomalies_file_exists_with_findings(self, tmp_path):
        reviewer = self._make(tmp_path, with_anomalies_file=True, anomalies_content=[{"id": 1}])
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 6: Anomaly Detection - PASS" in report

    def test_pass_when_anomalies_file_exists_but_empty(self, tmp_path):
        """Zero anomalies on clean data is valid — detector ran, nothing found."""
        reviewer = self._make(tmp_path, with_anomalies_file=True, anomalies_content=[])
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 6: Anomaly Detection - PASS" in report

    def test_fail_when_anomalies_file_missing(self, tmp_path):
        """Missing file means detector did not run."""
        reviewer = self._make(tmp_path, with_anomalies_file=False)
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "### Stage 6: Anomaly Detection - FAIL" in report

    def test_report_shows_detector_ran_true(self, tmp_path):
        reviewer = self._make(tmp_path, with_anomalies_file=True, anomalies_content=[])
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "Detector ran: True" in report

    def test_report_notes_zero_is_valid(self, tmp_path):
        reviewer = self._make(tmp_path, with_anomalies_file=True, anomalies_content=[])
        reviewer.generate_final_report()
        report = (tmp_path / "output" / "final_system_report.md").read_text(encoding="utf-8")
        assert "zero is valid on clean data" in report
